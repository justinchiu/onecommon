import copy
import os
import random
from collections import namedtuple
import tqdm

import numpy as np
import torch

from more_itertools.recipes import all_equal

from corpora.data import Dictionary, read_lines, get_tag


ReferenceRaw = namedtuple(
    "ReferenceRaw",
    "input_vals word_idxs referent_idxs output_idx \
    scenario_id real_ids partner_real_ids \
    agent chat_id \
    partner_referent_idxs partner_referent_our_view_idxs \
    referent_disagreements partner_referent_disagreements \
    non_pronoun_referent_idxs is_augmented".split()
)


ReferenceInstance = namedtuple(
    "ReferenceInstance",
    "ctx inpt tgt ref_inpt ref_tgt sel_tgt scenario_ids real_ids partner_real_ids \
    agents chat_ids sel_idxs lens \
    partner_ref_inpt partner_ref_tgt_our_view partner_num_markables \
    non_pronoun_ref_inpt non_pronoun_ref_tgt".split()
)

PRONOUNS = {'it', 'this', 'that', 'those', 'them', 'they'}

def process_referents(batch_of_referents, max_mentions=None):
    ref_inpt = []
    ref_tgt = []
    num_markables = []
    for j in range(len(batch_of_referents)):
        this_num_markables = len(batch_of_referents[j]) // 10
        if max_mentions is not None:
            this_num_markables = min(this_num_markables, max_mentions)
        num_markables.append(this_num_markables)
        if this_num_markables == 0:
            ref_inpt.append(torch.zeros(0,3).long())
            ref_tgt.append(torch.zeros(0,7).long())
        else:
            _ref_inpt = []
            _ref_tgt = []
            for k in range(this_num_markables):
                _ref_inpt.append(batch_of_referents[j][10 * k: 10 * k + 3])
                _ref_tgt.append(batch_of_referents[j][10 * k + 3: 10 * (k + 1)])
            ref_inpt.append(torch.tensor(_ref_inpt).long())
            ref_tgt.append(torch.tensor(_ref_tgt).long())
    if all(nm == 0 for nm in num_markables):
        ref_inpt = None
        ref_tgt = None
    else:
        ref_inpt = torch.nn.utils.rnn.pad_sequence(ref_inpt, batch_first=True)
        ref_tgt = torch.nn.utils.rnn.pad_sequence(ref_tgt, batch_first=True)
        assert ref_tgt.dim() == 3
        assert ref_inpt.dim() == 3
        for ix, nm in enumerate(num_markables):
            assert (ref_inpt[ix,nm:,:] == 0).all()
            assert (ref_tgt[ix,nm:,:] == 0).all()
    return ref_inpt, ref_tgt, torch.tensor(num_markables).long()

class ReferenceCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    SPATIAL_REPLACEMENTS = {
        'above': 'below',
        'below': 'above',
        'left': 'right',
        'right': 'left',
        'top': 'bottom',
        'bottom': 'top',
        'high': 'low',
        'low': 'high',
        'lower': 'higher',
        'higher': 'lower',
        'lowest': 'highest',
        'highest': 'lowest',
    }

    def __init__(self, domain, path, freq_cutoff=2, train='train_reference.txt',
                 valid='valid_reference.txt', test='test_reference.txt', verbose=False, word_dict=None,
                 max_instances_per_split=None, max_mentions_per_utterance=None, crosstalk_split=None,
                 spatial_data_augmentation_on_train=False,
                 ):
        self.verbose = verbose
        self.max_mentions_per_utterance = max_mentions_per_utterance
        self.crosstalk_split = crosstalk_split
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.spatial_replacements = {
            self.word_dict.word2idx[k]: self.word_dict.word2idx[v]
            for k, v in self.SPATIAL_REPLACEMENTS.items()
            if k in self.word_dict.word2idx and v in self.word_dict.word2idx
        }

        print("freq cutoff: {}".format(freq_cutoff))
        print("vocab size: {}".format(len(self.word_dict)))

        self.train = self.tokenize(os.path.join(path, train), max_instances_per_split, crosstalk_split,
                                   spatial_data_augmentation=spatial_data_augmentation_on_train) if train else []
        self.valid = self.tokenize(os.path.join(path, valid), max_instances_per_split, crosstalk_split) if valid else []
        self.test = self.tokenize(os.path.join(path, test), max_instances_per_split, crosstalk_split) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name, max_instances_per_split=None, crosstalk_split=None, spatial_data_augmentation=False):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)
        if max_instances_per_split is not None:
            lines = lines[:max_instances_per_split]

        unk = self.word_dict.get_idx('<unk>')
        # instances_by_chat_id = {}
        dataset, total, unks = [], 0, 0
        for line in lines:
        # for line in tqdm.tqdm(lines, ncols=80):
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            words = get_tag(tokens, 'dialogue')
            word_idxs = self.word_dict.w2i(words)
            referent_idxs = [int(val) for val in get_tag(tokens, 'referents')]
            partner_referent_idxs = [int(val) for val in get_tag(tokens, 'partner_referents')]
            partner_referent_our_view_idxs = [int(val) for val in get_tag(tokens, 'partner_referents_our_view')]
            output_idx = int(get_tag(tokens, 'output')[0])
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            partner_real_ids = get_tag(tokens, 'partner_real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]
            ref_disagreement = list(map(int, get_tag(tokens, 'referent_disagreements')))
            partner_ref_disagreement = list(map(int, get_tag(tokens, 'partner_referent_disagreements')))
            if crosstalk_split is not None:
                assert crosstalk_split in [0, 1]
                include = hash(chat_id) % 2 == crosstalk_split
            else:
                include = True
            non_pronoun_referent_idxs = [
                row
                for row in torch.tensor(referent_idxs).view(-1, 10)
                if not (set(words[row[0].item():row[1].item()+1]) & PRONOUNS)
            ]
            if non_pronoun_referent_idxs:
                non_pronoun_referent_idxs = torch.stack(non_pronoun_referent_idxs, 0).flatten().tolist()
            else:
                non_pronoun_referent_idxs = []
            instances = [ReferenceRaw(
                input_vals, word_idxs, referent_idxs, output_idx, scenario_id, real_ids, partner_real_ids, agent, chat_id,
                partner_referent_idxs, partner_referent_our_view_idxs, ref_disagreement, partner_ref_disagreement,
                non_pronoun_referent_idxs, False
            )]
            if spatial_data_augmentation:
                aug_input_vals = torch.tensor(input_vals).clone().view(7,4)
                aug_input_vals[:,:2] *= -1
                aug_input_vals = aug_input_vals.flatten().tolist()
                aug_word_idxs = [self.spatial_replacements.get(ix, ix) for ix in word_idxs]
                instances.append(ReferenceRaw(
                    aug_input_vals, aug_word_idxs, referent_idxs, output_idx, scenario_id, real_ids, partner_real_ids,
                    agent, chat_id, partner_referent_idxs, partner_referent_our_view_idxs, ref_disagreement, partner_ref_disagreement,
                    non_pronoun_referent_idxs, True
                ))
            if include:
                for instance in instances:
                    dataset.append(instance)
                    # compute statistics
                    total += len(word_idxs)
                    unks += np.count_nonzero([idx == unk for idx in word_idxs])

            # if chat_id not in instances_by_chat_id:
            #     instances_by_chat_id[chat_id] = []
            # instances_by_chat_id[chat_id].append(instance)

        if self.verbose:
            print('dataset %s, %d instances, total %d, unks %s, ratio %0.2f%%' % (
                file_name, len(dataset), total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        # NOT USED. see corpora/reference_sentence.py: ReferenceSentenceCorpus
        """Splits given dataset into batches."""
        if self.max_mentions_per_utterance is not None:
            raise NotImplementedError("--max_mentions_per_utterance for ReferenceCorpus")
        if shuffle:
            random.shuffle(dataset)

        # sort by markable length and pad
        dataset.sort(key=lambda x: len(x[2]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0, # for all utterances
            'nonpadn': 0, # for all utterances
        }

        i = 0
        pbar = tqdm.tqdm(total=len(dataset), ncols=80)
        while i < len(dataset):
            markable_length = len(dataset[i][2])

            ctxs, dials, refs, sels, scenario_ids, real_ids, partner_real_ids, agents, chat_ids, sel_idxs = [], [], [], [], [], [], [], [], [], []
            non_pronoun_refs = []
            lens = []
            partner_refs = []
            partner_refs_our_view = []

            for _ in range(bsz):
                if i >= len(dataset) or len(dataset[i][2]) != markable_length:
                    break
                ctxs.append(dataset[i][0])
                # deepcopy to prevent any padding issues with repeated calls
                dials.append(copy.deepcopy(dataset[i][1]))
                # dials.append(dataset[i][1])
                refs.append(dataset[i][2])
                sels.append(dataset[i][3])
                scenario_ids.append(dataset[i][4])
                real_ids.append(dataset[i][5])
                partner_real_ids.append(dataset[i][5])
                agents.append(dataset[i][7])
                chat_ids.append(dataset[i][8])
                sel_idxs.append(len(dataset[i][1]) - 1)
                partner_refs.append(dataset[i].partner_referent_idxs)
                partner_refs_our_view.append(dataset[i].partner_referent_our_view_idxs)
                non_pronoun_refs.append(dataset[i].non_pronoun_referent_idxs)
                i += 1
                pbar.update(1)

            # the longest dialogue in the batch
            max_len = max([len(dial) for dial in dials])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(dials)):
                stats['n'] += max_len
                stats['nonpadn'] += len(dials[j])
                # one additional pad
                dials[j] += [pad] * (max_len - len(dials[j]) + 1)
                lens.append(len(dials[j]))

            # construct tensor for context (bsz, num_ent * dim_ent)
            ctx = torch.Tensor(ctxs).float()

            # dialog data (seq_len, bsz)
            data = torch.Tensor(dials).long().transpose(0, 1).contiguous()

            # construct tensor for reference target
            num_markables = int(markable_length / 10)

            ref_inpt, ref_tgt, num_markables_by_sent = process_referents(refs)
            assert all(x.item() == num_markables for x in num_markables_by_sent)

            partner_ref_inpt, partner_ref_our_view_tgt, partner_num_markables_by_sent = process_referents(partner_refs_our_view)
            partner_ref_inpt_, partner_ref_tgt, _ = process_referents(partner_refs)
            assert torch.allclose(partner_ref_inpt, partner_ref_inpt_)

            non_pronoun_ref_inpt, non_pronoun_ref_tgt, _ = process_referents(non_pronoun_refs)

            # construct tensor for selection target
            sel_tgt = torch.Tensor(sels).long()
            if device is not None:
                ctx = ctx.to(device)
                data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            sel_idxs = torch.Tensor(sel_idxs).long()

            batches.append(ReferenceInstance(
                ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt,
                scenario_ids, real_ids, partner_real_ids, agents, chat_ids, sel_idxs, lens,
                partner_ref_inpt, partner_ref_our_view_tgt, partner_num_markables_by_sent,
                non_pronoun_ref_inpt, non_pronoun_ref_tgt
            ))
        pbar.close()

        if shuffle:
            random.shuffle(batches)

        return batches, stats
