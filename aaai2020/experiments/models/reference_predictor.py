import string

import pyro
import torch
from torch import nn
from torch.autograd import Variable

from models.attention_layers import StructuredTemporalAttentionLayer, StructuredAttentionLayer
from models.utils import bit_to_int_array, int_to_bit_array


class ReferencePredictor(object):
    def __init__(self, args):
        self.args = args
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def empty_stats(self):
        return {
            'correct': 0,
            'num_dots': 0,
            'gold_positive': 0,
            'pred_positive': 0,
            'true_positive': 0,
            'exact_match_num': 0,
            'exact_match_denom': 0,
        }

    def compute_stats(self, ref_mask, ref_tgt, ref_tgt_ix=None, ref_pred=None, ref_pred_ix=None, sum=True,
                      by_num_markables=False, num_markables=None):
        assert ref_pred is not None or ref_pred_ix is not None

        num_dots = ref_tgt.size(-1)
        if ref_tgt_ix is None:
            ref_tgt_ix = bit_to_int_array(ref_tgt.long())
        assert ref_tgt_ix.max().item() <= 2 ** num_dots

        if ref_pred is None:
            ref_pred = int_to_bit_array(ref_pred_ix, num_bits=num_dots)

        if ref_pred_ix is None:
            ref_pred_ix = bit_to_int_array(ref_pred.long())

        ref_mask_instance_level = (ref_mask.sum(-1) > 0).float()

        # N x bsz
        ref_exact_matches = ((ref_tgt_ix == ref_pred_ix) & ref_mask_instance_level.bool())

        stats = {
            'correct': ((ref_pred == ref_tgt.long()) * ref_mask.byte()),
            'num_dots': ref_mask,
            'gold_positive': ref_tgt,
            'pred_positive': (ref_pred * ref_mask.byte()),
            'true_positive': (ref_pred & ref_tgt.bool()),
            'exact_match_num': ref_exact_matches.float(),
            'exact_match_denom': ref_mask_instance_level.float(),
        }

        if sum:
            stats = {
                k: v.sum().item()
                for k, v in stats.items()
            }
            assert stats['pred_positive'] >= stats['true_positive']
            assert stats['gold_positive'] >= stats['true_positive']

        if by_num_markables:
            assert num_markables is not None
            for num_markable in num_markables.unique():
                num_markable = num_markable.item()
                if num_markable == 0:
                    continue
                mask = num_markables == num_markable
                nm_stats = self.compute_stats(
                    ref_mask[:,mask], ref_tgt[:,mask], ref_tgt_ix[:,mask],
                    ref_pred[:,mask], ref_pred_ix[:,mask], sum=sum,
                    by_num_markables=False, num_markables=num_markables[mask],
                )
                for k, v in nm_stats.items():
                    stats[f'nm-{num_markable}_{k}'] = v

        return stats

    def preprocess(self, ref_tgt, num_markables):
        ref_tgt = Variable(ref_tgt)
        ref_mask = torch.zeros_like(ref_tgt)
        for i, nm in enumerate(num_markables):
            ref_mask[i, :nm, :] = 1

        # max(this_num_markables) x batch_size x num_dots
        ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
        # print(ref_tgt.size())
        ref_mask = torch.transpose(ref_mask, 0, 1).contiguous()
        return ref_tgt, ref_mask

    def _forward_non_temporal(self, ref_out, ref_tgt, ref_tgt_ix, ref_mask):
        ref_out_logits, ref_out_full, _ = ref_out
        del ref_out

        N, bsz, num_dots = ref_tgt.size()
        ref_mask_instance_level = (ref_mask.sum(-1) > 0).float()

        if self.args.structured_attention_marginalize:
            assert ref_tgt.size() == ref_out_logits.size()
            assert ref_tgt.size() == ref_mask.size()
            # print('ref_out size: {}'.format(ref_out.size()))
            # print('ref_tgt size: {}'.format(ref_tgt.size()))
            ref_loss = (self.crit(ref_out_logits, ref_tgt) * ref_mask.float()).sum()
            ref_pred = (ref_out_logits > 0).long()

            ref_pred_ix = bit_to_int_array(ref_pred.long())

        else:
            ref_tgt_reshape = ref_tgt.view(-1, num_dots)
            ref_out_reshape = ref_out_full.view(-1, 2**num_dots)
            N_bsz = ref_tgt_reshape.size(0)
            assert N_bsz == ref_out_reshape.size(0)

            ref_loss = -(ref_out_reshape[torch.arange(N_bsz), ref_tgt_ix.view(-1)] * ref_mask_instance_level.view(-1)).sum()
            # N x bsz
            ref_pred_ix = ref_out_reshape.argmax(-1).view(N, bsz)
            # N x bsz x num_dots
            ref_pred = int_to_bit_array(ref_pred_ix, num_bits=num_dots)
        return ref_loss, ref_pred, ref_pred_ix

    def _forward_temporal(self, ref_out, ref_tgt, ref_tgt_ix, ref_mask, num_markables):
        from torch_struct import LinearChainNoScanCRF

        marginal_log_probs, joint_log_probs, temporal_dist = ref_out
        assert isinstance(temporal_dist, LinearChainNoScanCRF)

        num_dots = ref_tgt.size(-1)
        # bsz x N x 2**num_dots x 2**num_dots
        ref_tgt_ix_transpose = ref_tgt_ix.transpose(0,1)
        bsz, N, *_ = ref_tgt_ix_transpose.size()

        ref_loss = 0
        ref_pred = torch.zeros(N, bsz, num_dots).long()
        ref_pred_ix = torch.zeros(N, bsz).long()

        has_multiple = num_markables > 1

        # collect for single markables
        if (~has_multiple).any():
            ref_loss_single, ref_pred_single, ref_pred_ix_single = self._forward_non_temporal(
                (marginal_log_probs[:,~has_multiple], joint_log_probs[:,~has_multiple], None),
                ref_tgt[:,~has_multiple], ref_tgt_ix[:,~has_multiple],
                ref_mask[:,~has_multiple]
            )
            ref_loss += ref_loss_single
            ref_pred[:,~has_multiple] = ref_pred_single
            ref_pred_ix[:,~has_multiple] = ref_pred_ix_single


        ref_pred_ix_transpose_multi, exp_num_dots = LinearChainNoScanCRF.struct.from_parts(temporal_dist.argmax)
        assert exp_num_dots == 2**num_dots

        # aggregate with single timestep info
        # N x bsz
        ref_pred_ix[:,has_multiple] = ref_pred_ix_transpose_multi.transpose(0,1).contiguous()[:,has_multiple]
        ref_pred[:,has_multiple] = int_to_bit_array(ref_pred_ix[:, has_multiple], num_bits=num_dots)

        gold_parts = LinearChainNoScanCRF.struct.to_parts(
            ref_tgt_ix_transpose, 2**num_dots, lengths=num_markables
        )
        if self.args.structured_temporal_attention_training == 'likelihood':
            log_likelihoods = temporal_dist.log_prob(gold_parts)
            losses = -log_likelihoods
        elif self.args.structured_temporal_attention_training == 'max_margin':
            # TODO: need to do a loss-augmented decode
            raise NotImplementedError("max_margin")
            gold_scores = temporal_dist.score(gold_parts)
            pred_scores = temporal_dist.score(temporal_dist.argmax)
            # hinge loss with a structured margin
            errors = (ref_mask * (ref_pred != ref_tgt)).sum(0).sum(-1)
            losses = torch.max((pred_scores - gold_scores + errors), torch.zeros_like(gold_scores))
            # print(f"gold_scores: {gold_scores}")
            # print(f"pred_scores: {pred_scores}")
            # print(f"losses: {losses}")
        else:
            raise NotImplementedError("--structured_temporal_attention_training={}".format(self.args.structured_temporal_attention_training))

        ref_loss_multi = (losses * has_multiple).sum()
        ref_loss += ref_loss_multi
        return ref_loss, ref_pred, ref_pred_ix

    def forward(self, ref_inpt, ref_tgt, ref_out, num_markables, by_num_markables=False):
        if ref_inpt is None or ref_out is None:
            return None, None, self.empty_stats()

        ref_tgt, ref_mask = self.preprocess(ref_tgt, num_markables)

        # N: max(this_num_markables)
        N, bsz, num_dots = ref_tgt.size()

        # N x bsz
        ref_tgt_ix = bit_to_int_array(ref_tgt.long())
        assert ref_tgt_ix.max().item() <= 2 ** num_dots

        if ref_out[2] is None:
            ref_loss, ref_pred, ref_pred_ix = self._forward_non_temporal(ref_out, ref_tgt, ref_tgt_ix, ref_mask)
        else:
            ref_loss, ref_pred, ref_pred_ix = self._forward_temporal(ref_out, ref_tgt, ref_tgt_ix, ref_mask, num_markables)

        stats = self.compute_stats(ref_mask, ref_tgt, ref_tgt_ix, ref_pred, ref_pred_ix,
                                   by_num_markables=by_num_markables, num_markables=num_markables)

        return ref_loss, ref_pred, stats


class PragmaticReferencePredictor(ReferencePredictor):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--l1_sample', action='store_true', help='l1 listener should sample (otherwise top-k)')
        parser.add_argument('--l1_candidates', type=int, default=10, help='number of dot configurations for l1 to consider')
        parser.add_argument('--l1_speaker_weight', type=float, default=1.0, help='(1 - lambda) * l0_log_prob + (lambda) * s0_log_prob')
        parser.add_argument('--l1_oracle', action='store_true')
        parser.add_argument('--l1_exhaustive_single', action='store_true', help='search over all possible candidates for a single mention (should only compare nm-1 scores)')
        parser.add_argument('--l1_renormalize', action='store_true', help='normalize l1 to be over all candidates')

    def __init__(self, args):
        super().__init__(args)
        self._logit_to_full_einsum_str = None

    def marginal_logits_to_full_logits(self, logits):
        # TODO: refactor this
        return StructuredAttentionLayer.marginal_logits_to_full_logits(logits)

    def make_candidates(self, ref_out, num_markables):

        ref_out_logits, ref_out_full, ref_dist = ref_out

        N, bsz, num_dots = ref_out_logits.size()

        if self.args.l1_exhaustive_single:
            candidate_indices = torch.arange(2**num_dots).to(ref_out_logits.device)
            candidate_indices = candidate_indices.unsqueeze(0).repeat_interleave(N, 0)
            candidate_indices = candidate_indices.unsqueeze(1).repeat_interleave(bsz, 1)
            candidate_dots = int_to_bit_array(candidate_indices, num_bits=num_dots)
            candidate_l0_scores = torch.zeros(N, bsz, 2**num_dots).long().to(ref_out_logits.device)
            return candidate_indices, candidate_dots, candidate_l0_scores

        k = self.args.l1_candidates

        candidate_l0_scores = torch.zeros(N, bsz, k).to(ref_out_logits.device)
        candidate_indices = torch.zeros(N, bsz, k).long().to(ref_out_logits.device)

        use_temporal = num_markables > 1

        ## get scores and candidates for batch items where use_temporal == False
        if ref_out_full is None:
            ref_out_full = self.marginal_logits_to_full_logits(ref_out_logits).contiguous()

        if ref_dist is None and N > 1:
            ref_dist = StructuredTemporalAttentionLayer.make_distribution(
                ref_out_full, num_markables, transition_potentials=None
            )

        assert not self.args.l1_sample

        if N > 1:
            # k x bsz x N-1 x C x C
            candidate_edges = ref_dist.topk(k)
            # (k*bsz) x N
            candidates, C = ref_dist.struct.from_parts(candidate_edges.view((-1,) + candidate_edges.size()[2:]))
            assert C == 2**num_dots
            candidates = candidates.view(k, bsz, N)
            # N x bsz x k
            candidate_indices_multi = candidates.transpose(0,2)
            candidate_indices[:,use_temporal] = candidate_indices_multi[:,use_temporal]

            # k x bsz
            candidate_l0_scores_multi = ref_dist.log_prob(candidate_edges)
            # bsz x k
            candidate_l0_scores_multi = candidate_l0_scores_multi.transpose(0,1)
            # tile across the mention dimension N, which will allow us to argmax independently over the k dimension
            # to recover the joint argmax
            # N x bsz x k
            candidate_l0_scores_multi = candidate_l0_scores_multi.unsqueeze(0).repeat_interleave(N, dim=0)
            candidate_l0_scores[:,use_temporal] = candidate_l0_scores_multi[:,use_temporal]

        # num_mentions x bsz x
        assert ref_out_full.dim() == 2 + num_dots

        ref_out_full_reshape = ref_out_full.view(N, bsz, 2**num_dots)
        l0_log_probs = ref_out_full_reshape.log_softmax(dim=-1)

        if self.args.l1_sample:
            # sample without replacement
            scores = torch.distributions.Gumbel(l0_log_probs, scale=1.0).sample()
        else:
            scores = l0_log_probs
        tk = scores.topk(k=k, dim=-1)

        # N x bsz x k
        # l0_scores = tk.values
        candidate_indices_single = tk.indices
        candidate_l0_scores_single = l0_log_probs.gather(-1, candidate_indices_single)

        candidate_indices[:,~use_temporal] = candidate_indices_single[:,~use_temporal]

        # N x bsz x k
        candidate_l0_scores[:,~use_temporal] = candidate_l0_scores_single[:,~use_temporal]

        candidate_dots = int_to_bit_array(candidate_indices, num_bits=num_dots)
        candidate_dots = candidate_dots.view(N, bsz, k, num_dots)
        return candidate_indices, candidate_dots, candidate_l0_scores

    def forward(self, ref_inpt, ref_tgt, ref_out, num_markables, scoring_function, by_num_markables=False):
        if ref_inpt is None or ref_out is None:
            return None, None, self.empty_stats()

        ref_out_logits, ref_out_full, ref_dist = ref_out

        N, bsz, num_dots = ref_out_logits.size()
        ref_tgt_p, ref_mask = self.preprocess(ref_tgt, num_markables)

        candidate_indices, candidate_dots, candidate_l0_scores = self.make_candidates(ref_out, num_markables)

        k = candidate_dots.size(-1)

        if self.args.l1_oracle:
            # want to be able to take candidates over entire sequence of mentions
            assert self.args.structured_temporal_attention
            chosen = torch.zeros(N, bsz).long()
            for batch_index in range(bsz):
                # N x k x 7
                b_ref_mask = ref_mask[:, batch_index].unsqueeze(1).repeat_interleave(k, dim=1)
                # N x k x 7
                b_ref_tgt_p = ref_tgt_p[:, batch_index].unsqueeze(1).repeat_interleave(k, dim=1)
                # N x k x 7
                b_candidates = candidate_dots[:, batch_index]
                b_stats = self.compute_stats(b_ref_mask, b_ref_tgt_p, ref_pred=b_candidates, sum=False,
                                             by_num_markables=by_num_markables, num_markables=num_markables)

                # sum over num mentions and dots
                # k
                gold_positive = b_stats['gold_positive'].sum(0).sum(-1)
                # k
                true_positive = b_stats['true_positive'].sum(0).sum(-1)
                # k
                pred_positive = b_stats['pred_positive'].sum(0).sum(-1)

                exact_match_num = b_stats['exact_match_num'].sum(0)

                best_exact_match = (exact_match_num == exact_match_num.max(-1).values)

                # break ties with f1
                b_precision = true_positive / pred_positive
                b_precision[torch.isnan(b_precision)] = 0
                b_recall = true_positive / gold_positive
                b_recall[torch.isnan(b_recall)] = 0
                b_f1 = (2 * b_precision * b_recall) / (b_precision + b_recall)
                b_f1[torch.isnan(b_f1)] = 0

                b_f1_filtered = b_f1.clone()
                # choose among those entries that have exact match equal to the best, by setting other F1s to a negative value
                b_f1_filtered[~best_exact_match] = -1
                chosen_ix = b_f1_filtered.argmax(dim=-1)

                assert exact_match_num[chosen_ix].item() == exact_match_num.max(-1).values.item()

                chosen[:, batch_index] = chosen_ix

        else:
            # N x bsz x k
            # TODO: fix this so that we don't have to pass at inference time
            if vars(self.args).get('l1_loss_weight', 0.0) != 0.0:
                assert self.args.l1_renormalize

            # bsz x k
            # p(u | d_1, d_N)
            candidate_l1_scores = scoring_function(candidate_dots, normalize_over_candidates=self.args.l1_renormalize)
            candidate_l1_scores = candidate_l1_scores.unsqueeze(0).expand_as(candidate_l0_scores)
            # assert candidate_l1_scores.size() == candidate_l0_scores.size()

            lmbd = self.args.l1_speaker_weight
            # N x bsz x k
            joint_scores = (1 - lmbd) * candidate_l0_scores + lmbd * candidate_l1_scores

            # N x bsz with values [0..k-1]
            chosen = joint_scores.argmax(-1)

        chosen_indices = candidate_indices.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

        # convert indices to bits
        ref_pred = int_to_bit_array(chosen_indices, num_bits=num_dots)

        if self.args.l1_candidates == 1 and self.args.l1_speaker_weight == 0.0:
            ref_pred_l0 = super().forward(ref_inpt, ref_tgt, ref_out, num_markables)[1]
            assert (ref_pred == ref_pred_l0).all()

        stats = self.compute_stats(ref_mask, ref_tgt_p, ref_pred=ref_pred,
                                   by_num_markables=by_num_markables, num_markables=num_markables)
        ref_loss = 0.0
        return ref_loss, ref_pred, stats