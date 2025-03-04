import torch.nn.init

import corpora.markable
from corpora import data
from models.ctx_encoder import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx

#def prepare_sequence(seq, to_ix):
#    idxs = [to_ix[w] for w in seq]
#    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score, _ = vec.max(dim=1)
    max_score_broadcast = max_score.unsqueeze(1).expand(vec.size(0), vec.size(1))
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    corpus_ty = corpora.markable.MarkableCorpus

    def __init__(self, word_dict, tag_to_ix, embedding_dim, hidden_dim, device="cuda"):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        vocab_size = len(word_dict)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_dict = word_dict
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = device

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        if not hasattr(self, "device"):
            self.device = "cpu"
        return (
            torch.randn(2, 1, self.hidden_dim // 2, device=self.device),
            torch.randn(2, 1, self.hidden_dim // 2, device=self.device),
        )

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def detect_markables(self, dialog_tokens, dialog_text=None, device=None):
        word_indices = self.word_dict.w2i(dialog_tokens)
        word_indices = torch.tensor(word_indices, device=device, dtype=torch.long)

        score, tag_seq = self(word_indices)
        current_text = ""
        is_self = None
        end_idx = None
        for i, word in enumerate(word_indices):
            # loop over starting words, search for B tag
            if word.item() == self.word_dict.word2idx["YOU:"]:
                my_utterance = True
                # agent
                is_self = True
            elif word.item() == self.word_dict.word2idx["THEM:"]:
                my_utterance = False
                # 1 - agent
                is_self = False

            # it looks like sometimes EOS gets labelled as B
            definitely_not_entity = word.item() in self.word_dict.w2i(["<eos>", "<selection>"])

            if not definitely_not_entity and tag_seq[i].item() == self.tag_to_ix["B"]:
                start_idx = i
                for j in range(i + 1, len(tag_seq)):
                    if tag_seq[j].item() != self.tag_to_ix["I"]:
                        end_idx = j - 1
                        break
                    if word_indices[j].item() in self.word_dict.w2i(["<eos>", "<selection>"]):
                        # it looks like sometimes EOS gets labelled as I
                        end_idx = j - 1
                        break
                for j in range(i + 1, len(tag_seq)):
                    # JUSTIN: this seems incorrect? im guessing you want end_uttr to be the end of the utterance
                    # original one first
                    #if tag_seq[j].item() in self.word_dict.w2i(["<eos>", "<selection>"]):
                    if word_indices[j].item() in self.word_dict.w2i(["<eos>", "<selection>"]):
                    #if (
                        #tag_seq[j].item() in self.word_dict.w2i(["<eos>", "<selection>"])
                        #or word_indices[j].item() in self.word_dict.w2i(["<eos>", "<selection>"])
                    #):
                        end_uttr = j
                        break
                #if end_idx is None:
                    #import pdb; pdb.set_trace()
                markable_start = len(current_text + " ")
                markable = {
                    'start': markable_start,
                    'end': len(current_text + " " + " ".join(dialog_tokens[start_idx:end_idx + 1])),
                    'is_self': is_self,
                    'text': " ".join(dialog_tokens[start_idx:end_idx + 1])
                }
                referent_inpt = (start_idx, end_idx, end_uttr)
                yield markable, referent_inpt

            if word.item() == self.word_dict.word2idx["YOU:"]:
                # current_text += "{}:".format(current_speaker)
                current_text += "X:"
            elif word.item() == self.word_dict.word2idx["THEM:"]:
                # current_text += "{}:".format(current_speaker)
                current_text += "X:"
            elif word.item() in self.word_dict.w2i(["<eos>", "<selection>"]):
                current_text += "\n"
            else:
                current_text += " " + self.word_dict.idx2word[word.item()]

        if dialog_text is not None:
            assert len(current_text) == len(dialog_text)

#####################################################################
# Create model (batch training)

"""
class BiLSTM_CRF(nn.Module):
    corpus_ty = data.MarkableCorpus

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.batch_size, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:,next_tag].unsqueeze(1).expand(self.batch_size, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from 
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(self.batch_size, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(self.batch_size)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).unsqueeze(0).expand(1, self.batch_size), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1,:], tags[i,:]] + feat[list(range(16)), tags[i + 1,:]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1,:]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((self.batch_size, self.tagset_size), -10000.)
        init_vvars[list(range(16)),self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[list(range(self.batch_size)), best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t).view(self.batch_size, -1) + feat)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var).tolist()
        path_score = terminal_var[list(range(self.batch_size)), best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = [bptrs_t[tag_id][i].item() for i, tag_id in zip(range(self.batch_size), best_tag_id)]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start[0] == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
"""
