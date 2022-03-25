import copy
import random
from collections import namedtuple

import numpy as np
import torch

import tqdm

from corpora.reference import ReferenceCorpus, ReferenceRaw, process_referents

# helper function for collapsing a batch of raw references,
# any of refs/partner_refs/next_partner_refs,
# processed from tags in corpora/reference.py:tokenize
# references should be a list of per-turn references (of which there can be multiple)
# each set of per-turn references is flattened
# need to return something of shape ????
def collapse(references):
    # dot_mentions: batch x 7
    rs = []
    for r in references:
        rs.append(
            np.array(r, dtype=bool).reshape((-1, 10))[:,3:].any(0)
            if r else np.zeros((7,), dtype=bool)
        )
    return np.stack(rs)

# [ # per dialogue
    # [ # per turn
        # multiple mentions
    # ]
# ]

ReferenceSentenceInstance = namedtuple(
    "ReferenceSentenceInstance",
    "ctx inpts tgts ref_inpt ref_tgt sel_tgt scenario_ids \
    real_ids partner_real_ids \
    id_intersection \
    agents chat_ids sel_idxs lens rev_idxs hid_idxs num_markables is_self \
    partner_ref_inpt partner_ref_tgt_our_view partner_num_markables \
    next_partner_ref_inpt next_partner_ref_tgt_our_view next_partner_num_markables \
    next_partner_ref_intersect_ref next_partner_ref_complement_ref \
    referent_disagreements partner_referent_disagreements partner_ref_tgt is_selection \
    non_pronoun_ref_inpt non_pronoun_ref_tgt non_pronoun_num_markables is_augmented".split()
)

class ReferenceSentenceCorpus(ReferenceCorpus):
    # based on code from https://github.com/facebookresearch/end-to-end-negotiator/

    def _split_referents(self, words, referent_idxs):
        stops = [self.word_dict.get_idx(w) for w in ['YOU:', 'THEM:']]
        sents, current = [], []
        all_refs, current_refs = [], []
        split_ref_indices = []
        split_ref_objs = []
        assert len(referent_idxs) % 10 == 0
        for k in range(0, len(referent_idxs), 10):
            split_ref_indices.append(referent_idxs[k:k + 3])
            split_ref_objs.append(referent_idxs[k + 3:k + 10])
        split_ref_indices = np.array(split_ref_indices)
        ref_ix = 0
        for w in words:
            if w in stops:
                while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
                    current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
                    ref_ix += 1
                split_ref_indices[ref_ix:] -= len(current)
                if len(current) > 0:
                    sents.append(current)
                    all_refs.append(current_refs)
                current = []
                current_refs = []
            current.append(w)
        if len(current) > 0:
            while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
                current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
                ref_ix += 1
            sents.append(current)
            all_refs.append(current_refs)
        assert ref_ix == len(split_ref_indices)
        assert sum(len(refs) for refs in all_refs) == len(referent_idxs)
        assert len(all_refs) == len(sents)
        return sents, all_refs

    def _split_into_sentences(self, dataset):
        sent_dataset = []
        for reference_raw in dataset:
            words = reference_raw.word_idxs
            sents, all_refs = self._split_referents(words, reference_raw.referent_idxs)

            sents_, all_partner_refs = self._split_referents(words, reference_raw.partner_referent_idxs)
            assert sents == sents_

            sents_, all_partner_refs_our_view = self._split_referents(words, reference_raw.partner_referent_our_view_idxs)
            assert sents == sents_

            sents_, all_non_pronoun_refs = self._split_referents(words, reference_raw.non_pronoun_referent_idxs)
            assert sents == sents_

            new_ref_raw = reference_raw._replace(
                word_idxs=sents,
                referent_idxs=all_refs,
                partner_referent_idxs=all_partner_refs,
                partner_referent_our_view_idxs=all_partner_refs_our_view,
                non_pronoun_referent_idxs=all_non_pronoun_refs,
            )
            sent_dataset.append(new_ref_raw)
        # Sort by number of sentences, and then markable length
        sent_dataset.sort(key=lambda x: (len(x[1]), len(x[2])))

        return sent_dataset

    def _make_reverse_idxs(self, inpts, lens):
        idxs = []
        for inpt, ln in zip(inpts, lens):
            idx = torch.Tensor(inpt.size(0), inpt.size(1), 1).long().fill_(-1)
            for i in range(inpt.size(1)):
                arngmt = torch.Tensor(inpt.size(0), 1, 1).long()
                for j in range(arngmt.size(0)):
                    arngmt[j][0][0] = j if j > ln[i] else ln[i] - j
                idx.narrow(1, i, 1).copy_(arngmt)
            idxs.append(idx)
        return idxs

    def _make_hidden_idxs(self, lens):
        idxs = []
        for s, ln in enumerate(lens):
            idx = torch.Tensor(1, ln.size(0), 1).long()
            for i in range(ln.size(0)):
                idx[0][i][0] = ln[i]
            idxs.append(idx)
        return idxs

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        dataset = self._split_into_sentences(dataset)
        pad_index = self.word_dict.get_idx('<pad>')
        you_index = self.word_dict.get_idx('YOU:')
        them_index = self.word_dict.get_idx('THEM:')

        batches = []
        stats = {
            'n': 0, # for all utterances
            'self_n': 0, # for only those utterances spoken by this agent
            'nonpadn': 0, # for all utterances
            'self_nonpadn': 0, # for only those utterances spoken by this agent
            'max_num_mentions': 0,
        }

        i = 0
        pbar = tqdm.tqdm(total=len(dataset), ncols=80)
        while i < len(dataset):
            dial_len = len(dataset[i][1])
            # markable_length = len(dataset[i][2])

            ctxs, dials, refs, sels, scenario_ids, real_ids, partner_real_ids, agents, chat_ids, sel_idxs = [], [], [], [], [], [], [], [], [], []

            partner_refs, partner_refs_our_view = [], []
            non_pronoun_refs = []

            # MBP
            next_partner_refs, next_partner_refs_our_view = [], []
            next_partner_refs_intersect_ref = []
            next_partner_refs_complement_ref = []
            id_intersections = []
            # / MBP

            ref_disagreements, partner_ref_disagreements = [], []

            is_augmented = []

            for _ in range(bsz):
                # if i >= len(dataset) or len(dataset[i][1]) != dial_len or len(dataset[i][2]) != markable_length:
                if i >= len(dataset) or len(dataset[i][1]) != dial_len:
                    break
                ctxs.append(dataset[i][0])
                # deepcopy to prevent any padding issues with repeated calls
                dials.append(copy.deepcopy(dataset[i][1]))
                # dials.append(dataset[i][1])
                # TODO: may need to deal with per-sentence refs
                refs.append(dataset[i][2])
                sels.append(dataset[i][3])
                scenario_ids.append(dataset[i][4])
                real_ids.append(dataset[i][5])
                partner_real_ids.append(dataset[i][6])
                agents.append(dataset[i][7])
                chat_ids.append(dataset[i][8])
                sel_idxs.append(len(dataset[i][1][-1]) - 1)
                partner_refs.append(dataset[i].partner_referent_idxs)
                partner_refs_our_view.append(dataset[i].partner_referent_our_view_idxs)
                # MBP
                # shift one over and append empty reference for last next reference
                next_partner_refs.append(
                    dataset[i].partner_referent_idxs[1:] + [[]]
                )
                next_partner_refs_our_view.append(
                    dataset[i].partner_referent_our_view_idxs[1:] + [[]]
                )
                # boolean response to whether they repeat mentions
                # list of per-turn collapsed references
                # multiple references possible per turn
                collapsed_ref = collapse(refs[-1])
                collapsed_resp = collapse(next_partner_refs_our_view[-1])
                next_partner_refs_intersect_ref.append(collapsed_ref * collapsed_resp)
                next_partner_refs_complement_ref.append(~collapsed_ref * collapsed_resp)
                #if (collapsed_ref * collapsed_resp).any():
                    #import pdb; pdb.set_trace()
                # TODO: MAKE LAST NEXT PARTNER MENTIONS = NONE AT LAST TURN IN DIAL
                # TODO: COUNT NUMBER OF REPEAT MENTIONS
                # static dot intersection
                id_intersections.append([x in dataset[i][6] for x in dataset[i][5]])
                # / MBP
                non_pronoun_refs.append(dataset[i].non_pronoun_referent_idxs)
                ref_disagreements.append(dataset[i].referent_disagreements)
                partner_ref_disagreements.append(dataset[i].partner_referent_disagreements)
                is_augmented.append(dataset[i].is_augmented)
                i += 1
                pbar.update(1)

            inpts, lens, tgts = [], [], []
            ref_inpts, ref_tgts, all_num_markables = [], [], []
            partner_ref_inpts, partner_ref_tgts_our_view, all_partner_num_markables = [], [], []
            partner_ref_tgts = []
            # MBP
            # dont forget to use id_intersections
            next_partner_ref_inpts, next_partner_ref_tgts_our_view, next_all_partner_num_markables = [], [], []
            next_partner_ref_tgts = []
            next_partner_ref_intersect_refs = []
            next_partner_ref_complement_refs = []
            # / MBP
            non_pronoun_ref_inpts, non_pronoun_ref_tgts, all_non_pronoun_num_markables = [], [], []
            is_self = []
            for s in range(dial_len):
                batch = []
                for dial in dials:
                    batch.append(dial[s])
                if s + 1 < dial_len:
                    # add YOU:/THEM: as the last tokens in order to connect sentences
                    for j in range(len(batch)):
                        batch[j].append(dials[j][s + 1][0])
                else:
                    # add <pad> after <selection>
                    for j in range(len(batch)):
                        batch[j].append(pad_index)

                max_len = max([len(sent) for sent in batch])
                ln = torch.zeros(len(batch)).long()
                this_is_self = torch.zeros(len(batch)).bool()
                for j in range(len(batch)):
                    stats['n'] += max_len
                    # the last character should be either <pad> or YOU: or THEM:
                    # TODO: why was this minus 2?
                    utt_length = len(batch[j]) - 1
                    stats['nonpadn'] += utt_length
                    assert batch[j][-1] in [pad_index, you_index, them_index]
                    assert batch[j][-2] != pad_index
                    if batch[j][0] == you_index:
                        stats['self_n'] += max_len
                        stats['self_nonpadn'] += utt_length
                        this_is_self[j] = True
                    else:
                        this_is_self[j] = False
                        assert batch[j][0] == them_index
                    ln[j] = utt_length
                    batch[j] += [pad_index] * (max_len - len(batch[j]))
                is_self.append(this_is_self)
                sent = torch.Tensor(batch).long().transpose(0, 1).contiguous()
                inpt = sent.narrow(0, 0, sent.size(0) - 1)
                tgt = sent.narrow(0, 1, sent.size(0) - 1).view(-1)
                inpts.append(inpt)
                lens.append(ln)
                tgts.append(tgt)

                ref_inpt, ref_tgt, num_markables = process_referents(
                    [this_refs[s] for this_refs in refs],
                    max_mentions=self.max_mentions_per_utterance
                )
                if ref_tgt is not None:
                    stats['max_num_mentions'] = max(stats['max_num_mentions'], ref_tgt.size(1))
                ref_inpts.append(ref_inpt)
                ref_tgts.append(ref_tgt)
                all_num_markables.append(num_markables)

                if ref_inpt is not None:
                    for ix in range(ref_inpt.size(0)):
                        if num_markables[ix] > 0:
                            sentence = ref_inpt[ix,:num_markables[ix],2]
                            assert torch.all(sentence == sentence[0])

                partner_ref_inpt, partner_ref_tgt_our_view, partner_num_markables = process_referents(
                    [this_refs[s] for this_refs in partner_refs_our_view],
                    max_mentions=self.max_mentions_per_utterance,
                )
                partner_ref_inpt_, partner_ref_tgt, _ = process_referents(
                    [this_refs[s] for this_refs in partner_refs],
                    max_mentions=self.max_mentions_per_utterance,
                )
                assert (partner_ref_inpt is None and partner_ref_inpt_ is None) or torch.allclose(partner_ref_inpt, partner_ref_inpt_)

                partner_ref_inpts.append(partner_ref_inpt)
                partner_ref_tgts_our_view.append(partner_ref_tgt_our_view)
                partner_ref_tgts.append(partner_ref_tgt)
                all_partner_num_markables.append(partner_num_markables)

                # MBP
                # process references into RNN-ingestible format
                next_partner_ref_inpt, next_partner_ref_tgt_our_view, next_partner_num_markables = process_referents(
                    [this_refs[s] for this_refs in next_partner_refs_our_view],
                    max_mentions=self.max_mentions_per_utterance,
                )
                next_partner_ref_inpt_, next_partner_ref_tgt, _ = process_referents(
                    [this_refs[s] for this_refs in next_partner_refs],
                    max_mentions=self.max_mentions_per_utterance,
                )
                assert (
                    next_partner_ref_inpt is None
                    and next_partner_ref_inpt_ is None
                ) or torch.allclose(next_partner_ref_inpt, next_partner_ref_inpt_)

                next_partner_ref_inpts.append(next_partner_ref_inpt)
                next_partner_ref_tgts_our_view.append(next_partner_ref_tgt_our_view)
                next_partner_ref_tgts.append(next_partner_ref_tgt)
                next_all_partner_num_markables.append(next_partner_num_markables)

                # find intersection and complement with partner response and our utt
                next_partner_ref_intersect_refs.append([
                    x[s] for x in next_partner_refs_intersect_ref
                ])
                next_partner_ref_complement_refs.append([
                    x[s] for x in next_partner_refs_complement_ref
                ])
                # / MBP

                assert self.max_mentions_per_utterance is None
                non_pronoun_ref_inpt, non_pronoun_ref_tgt, non_pronoun_num_markables = process_referents(
                    [this_refs[s] for this_refs in non_pronoun_refs]
                )
                non_pronoun_ref_inpts.append(non_pronoun_ref_inpt)
                non_pronoun_ref_tgts.append(non_pronoun_ref_tgt)
                all_non_pronoun_num_markables.append(non_pronoun_num_markables)

            # # pad all the dialogues to match the longest dialogue
            # for j in range(len(dials)):
            #     stats['n'] += max_len
            #     stats['nonpadn'] += len(dials[j])
            #     # one additional pad
            #     dials[j] += [pad] * (max_len - len(dials[j]) + 1)

            # construct tensor for context (bsz, num_ent * dim_ent)
            ctx = torch.Tensor(ctxs).float()

            # dialog data (seq_len, bsz)
            # data = torch.Tensor(dials).long().transpose(0, 1).contiguous()

            # rev_idxs = self._make_reverse_idxs(inpts, lens)
            # hid_idxs = self._make_hidden_idxs(lens)
            rev_idxs = None
            hid_idxs = None

            # construct tensor for selection target
            sel_tgt = torch.Tensor(sels).long()
            if device is not None:
                ctx = ctx.to(device)
                # data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # # construct tensor for input and target
            # inpt = data.narrow(0, 0, data.size(0) - 1)
            # tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            sel_idxs = torch.Tensor(sel_idxs).long()

            assert len(inpts) == len(ref_inpts)

            is_selection = [
                # bsz is maximum batch size, so read the actual batch size off the inpt tensor
                torch.Tensor([ix == len(inpts) - 1 for _ in range(inpts[0].size(-1))]).bool().to(device)
                for ix in range(len(inpts))
            ]

            is_augmented = torch.tensor(is_augmented).bool()

            batches.append(ReferenceSentenceInstance(
                ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt,
                scenario_ids, real_ids, partner_real_ids,
                torch.tensor(id_intersections),
                agents, chat_ids, sel_idxs,
                lens, rev_idxs, hid_idxs, all_num_markables, is_self,
                partner_ref_inpts, partner_ref_tgts_our_view, all_partner_num_markables,
                next_partner_ref_inpts, next_partner_ref_tgts_our_view, next_all_partner_num_markables, # MBP
                torch.tensor(next_partner_ref_intersect_refs), # MBP
                torch.tensor(next_partner_ref_complement_refs), # MBP
                ref_disagreements, partner_ref_disagreements,
                partner_ref_tgts, is_selection,
                non_pronoun_ref_inpts, non_pronoun_ref_tgts, all_non_pronoun_num_markables,
                is_augmented,
            ))

        pbar.close()

        if shuffle:
            random.shuffle(batches)

        return batches, stats
