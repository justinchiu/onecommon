import os
from collections import namedtuple
import copy
import pprint
import time

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

from collections import defaultdict

import utils
from engines import EngineBase
from models.reference_predictor import ReferencePredictor
from models.utils import bit_to_int_array

CAN_CONFIRM_VALUES = {
    False: 0,
    True: 1,
    None: 2,
}

ForwardRet = namedtuple(
    "ForwardRet",
    ['lang_loss', 'unnormalized_lang_loss',
     'spatial_lang_loss', 'unnormalized_spatial_lang_loss', 'num_spatial_words',
     'word_attn_loss', 'feed_attn_loss',
     'num_words',
     'ref_loss', 'ref_stats',
     'sel_loss', 'sel_correct', 'sel_num_dots',
     # 'ref_gold_positive', 'ref_pred_positive', 'ref_true_positive', 'ref_num_dots', 'ref_em_num', 'ref_em_denom',
     # 'attn_ref_stats',
     'partner_ref_loss', 'partner_ref_stats',
     # 'partner_ref_correct', 'partner_ref_gold_positive', 'partner_ref_pred_positive', 'partner_ref_true_positive', 'partner_ref_num_dots', 'partner_ref_em_num', 'partner_ref_em_denom',
     'next_mention_loss', 'next_mention_stop_loss', 'next_mention_stats',
     # 'next_mention_correct', 'next_mention_gold_positive', 'next_mention_pred_positive',
     # 'next_mention_true_positive', 'next_mention_num_dots', 'next_mention_em_num', 'next_mention_em_denom',
     'is_selection_loss', 'is_selection_stats',
     'l1_loss',
     'relation_swapped_ref_loss', 'relation_swapped_partner_ref_loss'
     ],
)

def make_can_confirm_single(sentence_ix, is_self, previous_partner_num_markables, previous_partner_ref_tgts_our_view, resolution_strategy='any'):
    if sentence_ix == 0:
        has_previous_mention = torch.zeros_like(is_self)
        previous_resolves = torch.zeros_like(is_self)
    else:
        has_previous_mention = is_self & (previous_partner_num_markables > 0)
        if previous_partner_ref_tgts_our_view is None:
            previous_resolves = torch.zeros_like(has_previous_mention)
        else:
            mentions_resolve = previous_partner_ref_tgts_our_view.bool().any(-1)
            if resolution_strategy == 'any':
                previous_resolves = mentions_resolve.any(-1)
            elif resolution_strategy == 'all':
                previous_resolves = mentions_resolve.long().sum(-1) == previous_partner_num_markables
            elif resolution_strategy == 'half':
                previous_resolves = mentions_resolve.long().sum(-1) >= (previous_partner_num_markables.float() / 2)
            else:
                raise NotImplementedError(f"bad resolution strategy {resolution_strategy}")

    # values match boolean semantics
    assert (CAN_CONFIRM_VALUES[False] == 0) and (CAN_CONFIRM_VALUES[True] == 1)
    this_can_confirm = torch.where(has_previous_mention, previous_resolves.long(), torch.tensor(CAN_CONFIRM_VALUES[None]).long())
    return this_can_confirm

def make_can_confirm(is_self, partner_num_markables, partner_ref_tgts_our_view, resolution_strategy='any'):
    can_confirm = []
    for i, this_is_self in enumerate(is_self):
        this_can_confirm = make_can_confirm_single(i, this_is_self, partner_num_markables[i-1],
                                                   partner_ref_tgts_our_view[i-1],
                                                   resolution_strategy=resolution_strategy)
        can_confirm.append(this_can_confirm)
    return can_confirm

def add_loss_args(parser):
    group = parser.add_argument_group('loss')
    group.add_argument('--lang_weight', type=float, default=1.0,
                       help='language loss weight')
    group.add_argument('--spatial_lang_weight', type=float, default=0.0,
                       help='spatial language loss weight')
    group.add_argument('--ref_weight', type=float, default=1.0,
                       help='reference loss weight')
    group.add_argument('--partner_ref_weight', type=float, default=1.0,
                       help='partner reference loss weight')
    group.add_argument('--relation_swapped_ref_and_partner_ref_weight', type=float, default=1.0,
                       help='relation swapped ref + partner ref loss weight')
    group.add_argument('--sel_weight', type=float, default=1.0,
                       help='selection loss weight')
    group.add_argument('--next_mention_weight', type=float, default=1.0,
                       help='next mention loss weight')
    group.add_argument('--is_selection_weight', type=float, default=1.0,
                       help='is selection loss weight')
    group.add_argument('--next_mention_stop_weight', type=float, default=1.0,
                       help='next mention stop loss weight')
    group.add_argument('--next_mention_start_epoch', type=int,
                       help='only supervise next mention in this epoch onward (to allow pretraining)')
    group.add_argument('--selection_start_epoch', type=int,
                       help='only supervise selection in this epoch onward (to allow pretraining)')
    group.add_argument('--lang_only_self', action='store_true')

    group.add_argument('--max_mentions_in_generation_training',
                       type=int,
                       help='don\'t supervise/compute the language loss for more mentions than this')

    # l1 training
    group.add_argument('--l1_loss_weight', type=float, default=0.0,
                       help='weight for \log p(d | u) = p(d) \log p(u | d) - \log \sum_d\' \exp p(d\') p(u | d\')')
    group.add_argument('--l1_prior', choices=['uniform', 'next_mention'], default='uniform', help='how to parameterize p(d)')
    group.add_argument('--l1_normalizer_sampling', choices=['none', 'noised', 'uniform', 'next_mention'], default='none', help='how to sample candidates to compute the normalizer. none: exhaustive')
    group.add_argument('--l1_normalizer_sampling_candidates', type=int, help='how many candidates to sample when computing the normalizer')

    # these args only make sense if --lang_only_self is True
    group.add_argument('--word_attention_supervised', action='store_true')
    group.add_argument('--feed_attention_supervised', action='store_true')

    group.add_argument('--attention_supervision_method', choices=['kl', 'penalize_unmentioned'], default='kl')

    group.add_argument('--anneal_dot_recurrence', action='store_true')
    group.add_argument('--anneal_dot_recurrence_min_temperature', type=float, default=1e-2)

    group.add_argument('--dots_mentioned_no_pronouns', action='store_true')

    group.add_argument('--augmented_lang_only', action='store_true')
    group.add_argument('--relation_swap_augmentation', action='store_true')

def unwrap(loss):
    if loss is not None:
        return loss.item()
    else:
        return 0


def make_dots_mentioned(ref_tgt, args):
    assert ref_tgt.dim() == 3
    if args.only_first_mention:
        return ref_tgt[:, 0, :] > 0
    else:
        return ref_tgt.sum(1) > 0


def make_dots_mentioned_multi(refs, args, bsz, num_dots):
    dots_mentioned = []
    for ref_tgt in refs:
        if ref_tgt is None:
            dots_mentioned.append(torch.zeros(bsz, num_dots).bool())
            continue
        dots_mentioned.append(make_dots_mentioned(ref_tgt, args))
    return dots_mentioned

def make_dots_mentioned_per_ref_multi(refs, args, bsz, num_dots):
    dots_mentioned_per_ref = []
    for ref_tgt in refs:
        if ref_tgt is None:
            dots_mentioned_per_ref.append(torch.zeros(bsz, 0, num_dots).bool())
            continue
        dots_mentioned_per_ref.append(ref_tgt > 0)
    return dots_mentioned_per_ref

def add_selection_stats(single_sentence_is_selection, single_sentence_is_selection_outs, is_selection_stats):
    is_sel = single_sentence_is_selection
    is_sel_out = single_sentence_is_selection_outs

    is_sel_preds = is_sel_out > 0
    is_selection_stats['correct'] += (is_sel_preds == is_sel).sum()
    is_selection_stats['baseline_correct'] += (~is_sel).sum()
    # not really dots, but for compatibility with add_metrics
    is_selection_stats['num_dots'] += is_sel.flatten().size(0)
    is_selection_stats['gold_positive'] += is_sel.sum()
    is_selection_stats['pred_positive'] += is_sel_preds.sum()
    is_selection_stats['true_positive'] += (is_sel & is_sel_preds).sum()

def add_metrics(metric_dict_src, metric_dict_tgt, prefix, compute_em=True, compute_baseline=False):
    gold_positive = metric_dict_src['{}_gold_positive'.format(prefix)]
    pred_positive = metric_dict_src['{}_pred_positive'.format(prefix)]
    true_positive = metric_dict_src['{}_true_positive'.format(prefix)]
    correct = metric_dict_src['{}_correct'.format(prefix)]
    num_dots = metric_dict_src['{}_num_dots'.format(prefix)]

    precision = true_positive / pred_positive if pred_positive > 0 else 0
    recall = true_positive / gold_positive if gold_positive > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metric_dict_tgt['{}_accuracy'.format(prefix)] = correct / num_dots if num_dots > 0 else 0
    metric_dict_tgt['{}_precision'.format(prefix)] = precision
    metric_dict_tgt['{}_recall'.format(prefix)] = recall
    metric_dict_tgt['{}_f1'.format(prefix)] = f1

    if compute_em:
        em_num = metric_dict_src['{}_exact_match_num'.format(prefix)]
        em_denom = metric_dict_src['{}_exact_match_denom'.format(prefix)]
        metric_dict_tgt['{}_exact_match'.format(prefix)] = em_num / em_denom if em_denom > 0 else 0

    if compute_baseline:
        baseline_correct = metric_dict_src['{}_baseline_correct'.format(prefix)]
        metric_dict_tgt['{}_baseline_accuracy'.format(prefix)] = baseline_correct / num_dots if num_dots > 0 else 0

def flatten_metrics(metrics):
    flattened = {}
    for key, value in metrics.items():
        if key.endswith("_stats"):
            assert isinstance(value, dict)
            prefix = key[:-len("_stats")]
            for sub_k, sub_v in value.items():
                flattened[f'{prefix}_{sub_k}'] = sub_v
        else:
            assert not isinstance(value, dict)
            flattened[key] = value
    return flattened


class RnnReferenceEngine(EngineBase):
    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, model, args, verbose=False):
        super(RnnReferenceEngine, self).__init__(model, args, verbose)

    def _ref_loss(self, ref_inpt, ref_tgt, ref_out):
        if ref_inpt is not None and ref_out is not None:
            ref_out_logits, ref_out_full = ref_out
            assert ref_out_full is None
            ref_tgt = Variable(ref_tgt)
            ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
            ref_loss = self.ref_crit(ref_out_logits, ref_tgt)
            ref_pred = (ref_out_logits > 0).byte()
            ref_correct = (ref_pred.long() == ref_tgt.long()).sum().item()
            ref_num_dots = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)
            ref_gold_positive = ref_tgt.sum().item()
            ref_pred_positive = ref_pred.sum().item()
            ref_true_positive = (ref_pred & ref_tgt.byte()).sum().item()
            ref_em_num = (bit_to_int_array(ref_tgt.byte()) == bit_to_int_array(ref_pred)).sum()
            ref_em_denom = ref_tgt.size(0) * ref_tgt.size(1)
        else:
            ref_loss = None
            ref_pred = None
            ref_correct = 0
            ref_num_dots = 0
            ref_gold_positive = 0
            ref_pred_positive = 0
            ref_true_positive = 0
            ref_em_num = 0
            ref_em_denom = 0
        stats = {
            'correct': ref_correct,
            'num_dots': ref_num_dots,
            'gold_positive': ref_gold_positive,
            'pred_positive': ref_pred_positive,
            'true_positive': ref_true_positive,
            'exact_match_num': ref_em_num,
            'exact_match_denom': ref_em_denom,
        }
        return ref_loss, ref_pred, stats

    def _forward(self, batch, epoch, corpus, allow_relation_swap):
        assert not self.args.word_attention_supervised, 'this only makes sense for a hierarchical model, and --lang_only_self'
        assert not self.args.feed_attention_supervised, 'this only makes sense for a hierarchical model, and --lang_only_self'
        assert not self.args.mark_dots_mentioned, 'this only makes sense for a hierarchical model, and --lang_only_self'
        ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, _, _, _, _, sel_idx, lens, partner_ref_inpt, partner_ref_tgt_our_view, partner_num_markables = batch

        if allow_relation_swap and self.args.relation_swap_augmentation:
            raise NotImplementedError()

        ctx = Variable(ctx)
        inpt = Variable(inpt)
        bsz = ctx.size(0)
        if ref_inpt is not None:
            ref_inpt = Variable(ref_inpt)

        if ref_tgt is not None:
            dots_mentioned = make_dots_mentioned(ref_tgt, self.args)
        else:
            dots_mentioned = None

        raise NotImplementedError("dots_mentioned_per_ref, num_markables")

        # TODO: initialize is_selection from batch
        out, (ref_out, partner_ref_out), sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_out, (reader_lang_h, writer_lang_h) = \
            self.model.forward(
                ctx, inpt, ref_inpt, sel_idx,
                num_markables=None, # todo: fix this to be a vector of (bsz,) with constant value determined by the size of ref_tgt
                partner_num_markables=partner_num_markables,
                lens=None,
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                belief_constructor=None, partner_ref_inpt=partner_ref_inpt,
                is_selection=is_selection,
                can_confirm=None
            )

        tgt = Variable(tgt)
        sel_tgt = Variable(sel_tgt)
        lang_loss = self.crit(out, tgt)

        ref_loss, ref_predictions, ref_stats = self._ref_loss(
            ref_inpt, ref_tgt, ref_out
        )

        partner_ref_loss, partner_ref_predictions, partner_ref_stats = self._ref_loss(
            partner_ref_inpt, partner_ref_tgt_our_view, partner_ref_out
        )

        # second return value is None
        sel_out_logits, _ = sel_out

        sel_loss = self.sel_crit(sel_out_logits, sel_tgt)
        sel_correct = (sel_out_logits.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_out_logits.size(0)

        # TODO
        # attn_ref_stats = {}

        # TODO
        word_attn_loss = None
        feed_attn_loss = None

        return ForwardRet(
            lang_loss=lang_loss,
            unnormalized_lang_loss=0,
            word_attn_loss=word_attn_loss,
            feed_attn_loss=feed_attn_loss,
            num_words=1,
            ref_loss=ref_loss,
            ref_stats=ref_stats,
            sel_loss=sel_loss,
            sel_correct=sel_correct,
            sel_num_dots=sel_total,
            partner_ref_loss=partner_ref_loss,
            partner_ref_stats=partner_ref_stats,
            # TODO
            next_mention_loss=None,
            next_mention_stop_loss=None,
            next_mention_stats={},
            relation_swapped_ref_loss=None,
            relation_swapped_partner_ref_loss=None,
        )

    def train_batch(self, batch, epoch, corpus):
        forward_ret = self._forward(batch, epoch, corpus, allow_relation_swap=True)

        # default
        # TODO: sel_loss scaling varies based on whether lang_weight is positive
        # loss = None
        # if self.args.lang_weight > 0:
        #     loss = self.args.lang_weight * forward_ret.lang_loss
        #     if self.args.sel_weight > 0:
        #         loss += self.args.sel_weight * forward_ret.sel_loss
        #     if self.args.ref_weight > 0 and forward_ret.ref_loss is not None:
        #         loss += self.args.ref_weight * forward_ret.ref_loss
        # elif self.args.sel_weight > 0:
        #     loss = self.args.sel_weight * forward_ret.sel_loss / forward_ret.sel_total
        #     if self.args.ref_weight > 0 and forward_ret.ref_loss is not None:
        #         loss += self.args.ref_weight * forward_ret.ref_loss
        # elif self.args.ref_weight > 0 and forward_ret.ref_loss is not None:
        #     loss = self.args.ref_weight * forward_ret.ref_loss

        loss = 0.0
        if self.args.lang_weight > 0:
            loss += self.args.lang_weight * forward_ret.lang_loss
        if self.args.spatial_lang_weight > 0:
            loss += self.args.spatial_lang_weight * forward_ret.spatial_lang_loss
        if self.args.sel_weight > 0:
            if not self.args.selection_start_epoch or (epoch >= self.args.selection_start_epoch):
                loss += self.args.sel_weight * forward_ret.sel_loss
        if self.args.ref_weight > 0 and forward_ret.ref_loss is not None:
            loss += self.args.ref_weight * forward_ret.ref_loss
        if self.args.partner_ref_weight > 0 and forward_ret.partner_ref_loss is not None:
            loss += self.args.partner_ref_weight * forward_ret.partner_ref_loss
        if self.args.next_mention_weight > 0 and forward_ret.next_mention_loss is not None:
            if not self.args.next_mention_start_epoch or (epoch >= self.args.next_mention_start_epoch):
                loss += self.args.next_mention_weight * forward_ret.next_mention_loss
        if self.args.next_mention_stop_weight > 0 and forward_ret.next_mention_stop_loss is not None:
            if not self.args.next_mention_start_epoch or (epoch >= self.args.next_mention_start_epoch):
                loss += self.args.next_mention_stop_weight * forward_ret.next_mention_stop_loss

        if self.args.relation_swap_augmentation and self.args.relation_swapped_ref_and_partner_ref_weight > 0 and forward_ret.relation_swapped_ref_loss is not None:
            loss += self.args.relation_swapped_ref_and_partner_ref_weight * forward_ret.relation_swapped_ref_loss

        if self.args.relation_swap_augmentation and self.args.relation_swapped_ref_and_partner_ref_weight > 0 and forward_ret.relation_swapped_partner_ref_loss is not None:
            loss += self.args.relation_swapped_ref_and_partner_ref_weight * forward_ret.relation_swapped_partner_ref_loss

        if self.args.l1_loss_weight > 0 and forward_ret.l1_loss is not None:
            loss += self.args.l1_loss_weight * forward_ret.l1_loss

        if self.args.is_selection_weight > 0 and forward_ret.is_selection_loss is not None:
            loss += self.args.is_selection_weight * forward_ret.is_selection_loss

        if forward_ret.word_attn_loss is not None:
            loss = loss + forward_ret.word_attn_loss

        if forward_ret.feed_attn_loss is not None:
            loss = loss + forward_ret.feed_attn_loss

        if loss:
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()

        return forward_ret

    def valid_batch(self, batch, epoch, corpus):
        with torch.no_grad():
            return self._forward(batch, epoch, corpus, allow_relation_swap=False)

    def test_batch(self, batch, epoch, corpus):
        with torch.no_grad():
            return self._forward(batch, epoch, corpus, allow_relation_swap=False)

    def _pass(self, dataset, batch_fn, split_name, use_tqdm, epoch, corpus):
        start_time = time.time()

        metrics = {}

        for batch in tqdm.tqdm(dataset, ncols=80) if use_tqdm else dataset:
            # for batch in trainset:
            # lang_loss, ref_loss, ref_correct, ref_total, sel_loss, word_attn_loss, feed_attn_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = batch_fn(batch)
            forward_ret = batch_fn(batch, epoch, corpus)
            batch_metrics = flatten_metrics(forward_ret._asdict())
            metrics = utils.sum_dicts(metrics, batch_metrics)

        print("{} word_attn_loss: {:.4f}".format(split_name, metrics['word_attn_loss']))
        print("{} feed_attn_loss: {:.4f}".format(split_name, metrics['feed_attn_loss']))

        # pprint.pprint({'{}_{}'.format(name, k): v for k, v in total_attn_ref_stats.items()})

        time_elapsed = time.time() - start_time

        aggregate_metrics = {
            'lang_loss': metrics['lang_loss'] / len(dataset),
            'spatial_lang_loss': metrics['spatial_lang_loss'] / len(dataset),
            'ref_loss': metrics['ref_loss'] / len(dataset),
            'partner_ref_loss': metrics['partner_ref_loss'] / len(dataset),
            'next_mention_loss': metrics['next_mention_loss'] / len(dataset),
            'next_mention_stop_loss': metrics['next_mention_stop_loss'] / len(dataset),
            'select_loss': metrics['sel_loss'] / len(dataset),
            'is_selection_loss': metrics['is_selection_loss'] / len(dataset),
            # do select_accuracy here b/c we won't call add_metrics on select
            'select_accuracy': metrics['sel_correct'] / metrics['sel_num_dots'],
            'time': time_elapsed,
            'correct_ppl': np.exp(metrics['unnormalized_lang_loss'] / metrics['num_words']),
            'spatial_lang_ppl': np.exp(metrics['unnormalized_spatial_lang_loss'] / metrics['num_spatial_words']),
            'l1_loss': metrics['l1_loss'] / len(dataset),
            'relation_swapped_ref_loss': metrics['relation_swapped_ref_loss'] / len(dataset),
            'relation_swapped_partner_ref_loss': metrics['relation_swapped_partner_ref_loss'] / len(dataset),
        }
        add_metrics(metrics, aggregate_metrics, "ref")
        add_metrics(metrics, aggregate_metrics, "partner_ref")
        add_metrics(metrics, aggregate_metrics, "next_mention")
        if self.args.next_mention_prediction_type == 'multi_reference':
            add_metrics(metrics, aggregate_metrics, "next_mention_expanded")
        add_metrics(metrics, aggregate_metrics, "is_selection", compute_em=False, compute_baseline=True)
        # pprint.pprint(metrics)
        return aggregate_metrics

    def train_pass(self, trainset, trainset_stats, epoch, corpus):
        '''
        basic implementation of one training pass
        '''
        self.model.train()
        return self._pass(trainset, self.train_batch, "train", use_tqdm=True, epoch=epoch, corpus=corpus)

    def valid_pass(self, validset, validset_stats, epoch, corpus):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()
        return self._pass(validset, self.valid_batch, "val", use_tqdm=False, epoch=epoch, corpus=corpus)

    def iter(self, epoch, lr, traindata, validdata, corpus):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        if self.args.anneal_dot_recurrence:
            assert self.args.dot_recurrence
            temperatures = torch.linspace(1.0, self.args.anneal_dot_recurrence_min_temperature, self.args.max_epoch)
            self.model.dot_recurrence_weight_temperature = temperatures[epoch-1]

        train_metrics = self.train_pass(trainset, trainset_stats, epoch, corpus)
        valid_metrics = self.valid_pass(validset, validset_stats, epoch, corpus)

        if self.verbose:
            print('epoch %03d \t s/epoch %.2f \t lr %.2E' % (epoch, train_metrics['time'], lr))

            for split_name, metrics in [('train', train_metrics), ('valid', valid_metrics)]:
                metrics = metrics.copy()
                metrics['ppl(avg exp lang_loss)'] = np.exp(metrics['lang_loss'])
                metrics['lang_loss_unweighted'] = metrics['lang_loss']
                metrics['lang_loss'] *= self.args.lang_weight
                metrics['spatial_lang_loss'] *= self.args.spatial_lang_weight
                metrics['select_loss'] *= self.args.sel_weight
                metrics['ref_loss'] *= self.args.ref_weight
                metrics['partner_ref_loss'] *= self.args.partner_ref_weight
                metrics['next_mention_loss'] *= self.args.next_mention_weight
                metrics['next_mention_stop_loss'] *= self.args.next_mention_stop_weight
                metrics['is_selection_loss'] *= self.args.is_selection_weight
                metrics['l1_loss'] *= self.args.l1_loss_weight

                metrics['relation_swapped_ref_loss'] *= self.args.relation_swapped_ref_and_partner_ref_weight
                metrics['relation_swapped_partner_ref_loss'] *= self.args.relation_swapped_ref_and_partner_ref_weight

                quantities = [
                    ['lang_loss', 'ppl(avg exp lang_loss)', 'correct_ppl'],
                    ['spatial_lang_ppl'],
                    ['select_loss', 'select_accuracy'],
                    ['ref_loss', 'ref_accuracy', 'ref_precision', 'ref_recall', 'ref_f1', 'ref_exact_match'],
                    ['partner_ref_loss', 'partner_ref_accuracy', 'partner_ref_precision', 'partner_ref_recall',
                     'partner_ref_f1', 'partner_ref_exact_match'],
                    ['relation_swapped_ref_loss', 'relation_swapped_partner_ref_loss'],
                    ['next_mention_loss', 'next_mention_stop_loss',
                     'next_mention_accuracy', 'next_mention_precision', 'next_mention_recall',
                     'next_mention_f1', 'next_mention_exact_match'],
                    ['is_selection_baseline_accuracy'],
                    ['is_selection_loss', 'is_selection_accuracy', 'is_selection_precision', 'is_selection_recall', 'is_selection_f1'],
                ]
                if self.args.next_mention_prediction_type == 'multi_reference':
                    quantities.append(
                        ['next_mention_expanded_accuracy', 'next_mention_expanded_precision',
                         'next_mention_expanded_recall', 'next_mention_expanded_f1',
                         'next_mention_expanded_exact_match']
                    )
                quantities.append(['l1_loss'])
                for line_metrics in quantities:
                    print('epoch {:03d} \t '.format(epoch) + ' \t '.join(
                        ('%s_%s {%s:.4f}' % (split_name, metric, metric)).format(**metrics)
                        for metric in line_metrics
                    ))

            print()

        metrics = {
            'train': train_metrics,
            'valid': valid_metrics
        }

        if self.args.tensorboard_log:
            for split, split_metrics in metrics.items():
                for t, value in split_metrics.items():
                    tag = '{}_{}'.format(split, t)
                    self.logger.scalar_summary(tag, value, epoch)

            for tag, value in self.model.named_parameters():
                if value.grad is None:
                    continue
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                self.logger.histo_summary(
                    tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        return metrics

    def combine_loss(self, metrics):
        # return metrics['lang_loss'] * int(self.args.lang_weight > 0) \
        #        + metrics['select_loss'] * int(self.args.sel_weight > 0) \
        #        + metrics['ref_loss'] * int(self.args.ref_weight > 0) \
        #        + metrics['partner_ref_loss'] * int(self.args.partner_ref_weight > 0)

        rs_weights = self.args.relation_swapped_ref_and_partner_ref_weight if self.args.relation_swap_augmentation else 0.0

        return metrics['lang_loss'] * self.args.lang_weight \
               + metrics['spatial_lang_loss'] * self.args.spatial_lang_weight \
               + metrics['select_loss'] * self.args.sel_weight \
               + metrics['ref_loss'] * self.args.ref_weight \
               + metrics['partner_ref_loss'] * self.args.partner_ref_weight \
               + metrics['next_mention_loss'] * self.args.next_mention_weight \
               + metrics['next_mention_stop_loss'] * self.args.next_mention_stop_weight \
               + metrics['is_selection_loss'] * self.args.is_selection_weight \
               + metrics['l1_loss'] * self.args.l1_loss_weight \
               + metrics['relation_swapped_ref_loss'] * rs_weights \
               + metrics['relation_swapped_partner_ref_loss'] * rs_weights

    def train(self, corpus, model_filename_fn):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        validdata = corpus.valid_dataset(self.args.bsz)

        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            print("train set stats:")
            pprint.pprint(traindata[1])

            metrics = self.iter(epoch, self.opt.param_groups[0]["lr"], traindata, validdata, corpus)

            # valid_lang_loss, valid_select_loss, valid_reference_loss, valid_select_acc = self.iter(
            # )

            combined_valid_loss = self.combine_loss(metrics['valid'])

            if self.scheduler is not None:
                self.scheduler.step(combined_valid_loss)

            if combined_valid_loss < best_combined_valid_loss:
                print(
                    f"update best model -- valid combined: {combined_valid_loss:.4f}\t" +
                    '\t'.join(
                        f'{name} {metrics["valid"][name]:.4f}'
                        for name in ['lang_loss',
                                     'select_loss', 'select_accuracy',
                                     'ref_loss', 'partner_ref_loss',
                                     'next_mention_loss', 'next_mention_stop_loss',
                                     'is_selection_loss',
                                     'l1_loss',
                                     'relation_swapped_ref_loss', 'relation_swapped_partner_ref_loss',
                                     ]
                    )
                )
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

                utils.save_model(best_model, model_filename_fn('ep-{}'.format(epoch), 'th'), prefix_dir=None)
                utils.save_model(best_model.state_dict(), model_filename_fn('ep-{}'.format(epoch), 'stdict'), prefix_dir=None)

        return best_combined_valid_loss, best_model


class HierarchicalRnnReferenceEngine(RnnReferenceEngine):
    @classmethod
    def add_args(cls, parser):
        # don't need to call super because its arguments will already be registered by engines.add_engine_args
        pass

    def __init__(self, model, args, verbose=False):
        super(HierarchicalRnnReferenceEngine, self).__init__(model, args, verbose=verbose)
        self.ref_loss = ReferencePredictor(args)
        # TODO: make this less hacky, have all classes use ReferenceLoss?
        del self.ref_crit_no_reduce

    def _append_pad(self, inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables):
        # FAIR's e2e code had this because it was used in the latent clustering pre-training objective; we shouldn't need it
        bsz = inpts[0].size(1)
        pad = torch.Tensor(bsz).fill_(self.model.word_dict.get_idx('<pad>')).long()
        inpts.append(Variable(pad.unsqueeze(0)))
        ref_inpts.append(None)
        ref_tgts.append(None)
        num_markables.append(Variable(torch.zeros(bsz).long()))
        tgts.append(Variable(pad))
        lens.append(torch.Tensor(bsz).cpu().fill_(0).long())
        rev_idxs.append(torch.Tensor(1, bsz, 1).fill_(0).long())
        hid_idxs.append(torch.Tensor(1, bsz, 1).fill_(0).long())
        return inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables

    def _ref_loss(self, *args, **kwargs):
        return self.ref_loss.forward(*args, **kwargs)

    def _forward(self, batch, epoch, corpus, allow_relation_swap):
        if self.args.word_attention_supervised or self.args.feed_attention_supervised or self.args.mark_dots_mentioned:
            assert self.args.lang_only_self
        (
            ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, scenario_ids,
            real_ids, partner_real_ids, id_intersection, _, _,
            sel_idx, lens, rev_idxs, hid_idxs, num_markables, is_self,
            partner_ref_inpts, partner_ref_tgts_our_view, partner_num_markables,
            next_partner_ref_inpt, next_partner_ref_tgt_our_view, next_partner_num_markables, # MBP
            next_partner_ref_intersect_ref, next_partner_ref_complement_ref, # MBP
            ref_disagreements, partner_ref_disagreements, partner_ref_tgts_their_view,
            is_selection, non_pronoun_ref_inpts, non_pronoun_ref_tgts, non_pronoun_num_markables, is_augmented,
        ) = batch

        ctx = Variable(ctx)
        bsz = ctx.size(0)
        num_dots = int(ctx.size(1) / 4)
        assert num_dots == 7

        assert is_selection[0].size(0) == bsz

        if self.args.augmented_lang_only:
            non_lang_instance_mask = is_augmented
        else:
            non_lang_instance_mask = torch.zeros_like(is_augmented)

        inpts = [Variable(inpt) for inpt in inpts]
        ref_inpts = [Variable(ref_inpt) if ref_inpt is not None else None
                     for ref_inpt in ref_inpts]
        tgts = [Variable(tgt) for tgt in tgts]
        # rev_idxs = [Variable(idx) for idx in rev_idxs]
        # hid_idxs = [Variable(idx) for idx in hid_idxs]

        # inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables, = self._append_pad(inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables)

        # if single_sent_generation_beliefs is not None:
        #     generation_beliefs = [single_sent_generation_beliefs] * len(inpts)
        # else:
        #     generation_beliefs = None

        last_partner_ref_out = None

        if self.args.dots_mentioned_no_pronouns:
            dots_mentioned = make_dots_mentioned_multi(non_pronoun_ref_tgts, self.args, bsz, num_dots)
            dots_mentioned_per_ref = make_dots_mentioned_per_ref_multi(non_pronoun_ref_tgts, self.args, bsz, num_dots)
            dots_mentioned_num_markables = non_pronoun_num_markables
        else:
            dots_mentioned = make_dots_mentioned_multi(ref_tgts, self.args, bsz, num_dots)
            dots_mentioned_per_ref = make_dots_mentioned_per_ref_multi(ref_tgts, self.args, bsz, num_dots)
            dots_mentioned_num_markables = num_markables

        partner_dots_mentioned_our_view = make_dots_mentioned_multi(
            partner_ref_tgts_our_view, self.args, bsz, num_dots
        )
        partner_dots_mentioned_our_view_per_ref = make_dots_mentioned_per_ref_multi(
            partner_ref_tgts_our_view, self.args, bsz, num_dots
        )

        # TODO: fix module structure so we can import this up top without a circular import
        from engines.beliefs import BeliefConstructor
        belief_constructor = BeliefConstructor(
            self.args, epoch, bsz, num_dots, inpts, ref_tgts, partner_ref_tgts_our_view,
            real_ids, partner_real_ids, sel_tgt, is_self,
            partner_dots_mentioned_our_view, partner_dots_mentioned_our_view_per_ref,
            dots_mentioned, dots_mentioned_per_ref,
            ref_inpts, partner_ref_inpts,
            num_markables,
            partner_num_markables,
        )

        compute_l1_loss = self.args.l1_loss_weight > 0

        can_confirm = make_can_confirm(is_self, partner_num_markables, partner_ref_tgts_our_view, resolution_strategy=self.args.confirmations_resolution_strategy)

        state, outs, ref_outs_and_partner_ref_outs, sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_outs,\
        is_selection_outs, (reader_lang_hs, writer_lang_hs), l1_log_probs, next_mention_candidates,\
        relation_swapped_ref_outs, relation_swapped_partner_ref_outs, has_relation_swap = self.model.forward(
            ctx, inpts, ref_inpts, sel_idx,
            num_markables, partner_num_markables,
            lens,
            dots_mentioned, dots_mentioned_per_ref, dots_mentioned_num_markables,
            belief_constructor=belief_constructor,
            partner_ref_inpts=partner_ref_inpts,
            compute_l1_probs=compute_l1_loss,
            tgts=tgts,
            ref_tgts=ref_tgts, partner_ref_tgts=partner_ref_tgts_our_view,
            force_next_mention_num_markables=True,
            is_selection=is_selection,
            can_confirm=can_confirm,
            relation_swap=self.args.relation_swap_augmentation,
        )

        sel_tgt = Variable(sel_tgt)
        lang_losses = []
        spatial_lang_losses = []
        spatial_lang_num_words = []
        l1_losses = []
        assert len(inpts) == len(tgts) == len(outs) == len(lens)
        for i, (out, tgt) in enumerate(zip(outs, tgts)):
            # T x bsz
            loss = self.crit_no_reduce(out, tgt).view(-1, bsz)
            if self.args.max_mentions_in_generation_training is not None:
                assert self.args.lang_only_self
            if compute_l1_loss:
                assert self.args.lang_only_self

            spatial_mask = torch.BoolTensor([[ix.item() in corpus.spatial_replacements.keys() for ix in row]
                            for row in tgt.view(-1, bsz)]).to(loss.device)
            spatial_lang_loss = (loss * spatial_mask).sum()
            spatial_lang_losses.append(spatial_lang_loss)
            spatial_lang_num_words.append(spatial_mask.sum())
            if self.args.lang_only_self:
                # loss = loss * (this_is_self.unsqueeze(0).expand_as(loss))
                mask = is_self[i]
                if self.args.max_mentions_in_generation_training is not None:
                    mask = mask & (dots_mentioned_num_markables[i] <= self.args.max_mentions_in_generation_training)
                loss = loss * (mask.unsqueeze(0).expand_as(loss))

                l1_mask = mask & (dots_mentioned_num_markables[i] > 0)
                if compute_l1_loss and l1_log_probs[i] is not None and l1_mask.any():
                    # l1_scores[i]: max_num_mentions x bsz
                    normalizer = (dots_mentioned_num_markables[i] * l1_mask).sum()
                    assert normalizer.item() != 0
                    l1_losses.append(-(l1_log_probs[i] * l1_mask).sum() / normalizer)
                else:
                    l1_losses.append(torch.tensor(0.0))
            lang_losses.append(loss.sum())
        total_lens = sum(l.sum() for l in lens)

        # w1 w2 w3 ... <eos> YOU:
        # or
        # w1 w2 w3 ... <eos> THEM:
        # subtract 2 to remove <eos> and YOU/THEM:
        total_lens_no_eos = sum((l - 2).sum().item() for l in lens)

        unnormed_lang_loss = sum(lang_losses)
        lang_loss = unnormed_lang_loss / total_lens

        unnormed_spatial_lang_loss = sum(spatial_lang_losses)
        num_spatial_words = sum(spatial_lang_num_words)
        spatial_lang_loss = unnormed_spatial_lang_loss / num_spatial_words if num_spatial_words > 0 else torch.tensor(0.0)

        if l1_losses:
            l1_loss = torch.mean(torch.stack(l1_losses, -1), -1)
        else:
            l1_loss = None

        ref_losses = []
        # other keys and values will be added
        # ref_stats = {'num_dots': 0}
        ref_stats = defaultdict(lambda: 0.0)

        partner_ref_losses = []
        # other keys and values will be added
        # partner_ref_stats = {'num_dots': 0}
        partner_ref_stats = defaultdict(lambda: 0.0)

        rs_ref_stats = defaultdict(lambda: 0.0)
        rs_partner_ref_stats = defaultdict(lambda: 0.0)

        attn_ref_true_positive = 0
        attn_ref_total = 0
        attn_ref_gold_positive = 0
        attn_ref_pred_positive = 0

        word_attn_losses = []
        feed_attn_losses = []

        relation_swapped_ref_losses = []
        relation_swapped_partner_ref_losses = []

        assert len(ref_inpts) == len(ref_tgts) == len(num_markables)

        assert len(partner_ref_inpts) == len(partner_ref_tgts_our_view) == len(partner_num_markables)

        # TODO: just index into the lists; the safety check isn't worth it
        for sentence_ix, (ref_inpt, partner_ref_inpt, (ref_out, partner_ref_out), ref_tgt, partner_ref_tgt,\
            this_num_markables,this_partner_num_markables, this_ctx_attn_prob, this_feed_ctx_attn_prob, \
            this_dots_mentioned, this_dots_mentioned_per_ref, inpt, tgt) in enumerate(utils.safe_zip(
            ref_inpts, partner_ref_inpts, ref_outs_and_partner_ref_outs, ref_tgts, partner_ref_tgts_our_view,
            num_markables, partner_num_markables, ctx_attn_prob, feed_ctx_attn_prob,
            dots_mentioned, dots_mentioned_per_ref, inpts, tgts)
        ):
            if (this_num_markables == 0).all() or ref_tgt is None:
                continue
            assert max(this_num_markables) == ref_tgt.size(1)

            if (~non_lang_instance_mask).any():
                _ref_loss, _ref_pred, _ref_stats = self._ref_loss(
                    ref_inpt, ref_tgt, ref_out, this_num_markables,
                    mask=~non_lang_instance_mask
                )
                ref_losses.append(_ref_loss)
                ref_stats = utils.sum_dicts(ref_stats, _ref_stats)

                _partner_ref_loss, _partner_ref_pred, _partner_ref_stats = self._ref_loss(
                    partner_ref_inpt, partner_ref_tgt, partner_ref_out, this_partner_num_markables,
                    mask=~non_lang_instance_mask
                )
                partner_ref_stats = utils.sum_dicts(partner_ref_stats, _partner_ref_stats)
                if _partner_ref_loss is not None:
                    partner_ref_losses.append(_partner_ref_loss)

                if self.args.relation_swap_augmentation:
                    this_has_relation_swap = has_relation_swap[sentence_ix]
                    if this_has_relation_swap.any():
                        # rs_max_num_markables = this_num_markables[this_has_relation_swap].max()
                        # _rs_ref_loss, _, _rs_ref_stats = self._ref_loss(
                        #     ref_inpt[this_has_relation_swap][:,:rs_max_num_markables] if ref_inpt is not None else None,
                        #     (0 * ref_tgt[this_has_relation_swap][:,:rs_max_num_markables]) if ref_tgt is not None else None,
                        #     relation_swapped_ref_outs[sentence_ix],
                        #     this_num_markables[this_has_relation_swap],
                        #     mask=(~non_lang_instance_mask)[this_has_relation_swap]
                        # )
                        _rs_ref_loss, _, _rs_ref_stats = self._ref_loss(
                            ref_inpt,
                            (0 * ref_tgt) if ref_tgt is not None else None,
                            relation_swapped_ref_outs[sentence_ix],
                            this_num_markables,
                            mask=(~non_lang_instance_mask) & this_has_relation_swap
                        )
                        relation_swapped_ref_losses.append(_rs_ref_loss)
                        rs_ref_stats = utils.sum_dicts(rs_ref_stats, _rs_ref_stats)

                        # rs_max_partner_num_markables = this_partner_num_markables[this_has_relation_swap].max()
                        # _rs_partner_ref_loss, _, _rs_partner_ref_stats = self._ref_loss(
                        #     partner_ref_inpt[this_has_relation_swap][:,:rs_max_partner_num_markables] if partner_ref_inpt is not None else None,
                        #     (0 * partner_ref_tgt[this_has_relation_swap][:,:rs_max_partner_num_markables]) if partner_ref_tgt is not None else None,
                        #     relation_swapped_partner_ref_outs[sentence_ix],
                        #     this_partner_num_markables[this_has_relation_swap],
                        #     mask=(~non_lang_instance_mask)[this_has_relation_swap]
                        # )
                        _rs_partner_ref_loss, _, _rs_partner_ref_stats = self._ref_loss(
                            partner_ref_inpt,
                            (0 * partner_ref_tgt) if partner_ref_tgt is not None else None,
                            relation_swapped_partner_ref_outs[sentence_ix],
                            this_partner_num_markables,
                            mask=(~non_lang_instance_mask) & this_has_relation_swap
                        )
                        relation_swapped_partner_ref_losses.append(_rs_partner_ref_loss)
                        rs_partner_ref_stats = utils.sum_dicts(rs_partner_ref_stats, _rs_partner_ref_stats)

            if this_ctx_attn_prob is not None:
                # this_ctx_attn_prob: N x batch x num_dots
                tcap = this_ctx_attn_prob

                # (N*batch) x num_dots
                # tcap = this_ctx_attn_prob.view(-1, this_ctx_attn_prob.size(-1))

                # get the dots that receive the highest attention probs, up to p% of the mass
                top_attn, sorted_ix = tcap.sort(dim=-1, descending=True)
                non_attended = top_attn.cumsum(-1) > 0.75
                # from Thom Wolf's nucleus sampling implementation, https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
                # shift to also keep the first index above the threshold
                non_attended[..., 1:] = non_attended[..., :-1].clone()
                non_attended[..., 0] = 0
                attended = ~non_attended
                for batch_ix, nm in enumerate(this_num_markables):
                    for markable_ix in range(nm):
                        markable_start, markable_end, _ = ref_inpt[batch_ix, markable_ix]
                        if markable_end < markable_start:
                            continue
                        assert markable_start > 0  # these are indexes into inpt; we need to subtact one to get indices into tgt
                        pred_dots = set()
                        for t in range(markable_start, markable_end + 1):
                            # indices into the original tcap; i.e. dot indices
                            this_pred_dots = sorted_ix[t, batch_ix][attended[t - 1, batch_ix]]
                            pred_dots.update(set(this_pred_dots.cpu().detach().numpy()))
                        gold_pos = set(ref_tgt[batch_ix, markable_ix].nonzero().flatten().cpu().detach().numpy())
                        attn_ref_true_positive += len(pred_dots & gold_pos)
                        attn_ref_gold_positive += len(gold_pos)
                        attn_ref_total += num_dots
                        attn_ref_pred_positive += len(pred_dots)

                        if self.args.word_attention_supervised:
                            gold_dist = ref_tgt[batch_ix, markable_ix]
                            gold_dist = gold_dist / gold_dist.sum()
                            # num_locations x num_dots
                            referent_attention = this_ctx_attn_prob[markable_start - 1:markable_end, batch_ix]
                            # kl_div takes inputs as log probabilities, and target probabilities
                            if self.args.attention_supervision_method == 'kl':
                                attn_loss = torch.nn.functional.kl_div(
                                    referent_attention.log(),
                                    gold_dist.unsqueeze(0).expand_as(referent_attention),
                                    reduction='batchmean'
                                )
                            elif self.args.attention_supervision_method == 'penalize_unmentioned':
                                # attn_loss = referent_attention[gold_dist == 0].log().sum()
                                attn_loss = referent_attention[
                                    (gold_dist == 0).unsqueeze(0).expand_as(referent_attention)].sum()
                            else:
                                raise ValueError("invalid --attention_supervision_method {}".format(
                                    self.args.attention_supervision_method))

                            if attn_loss != attn_loss:
                                print("nan loss: {}\nreferent_attention: {} \t gold_dist: {}".format(attn_loss.item(),
                                                                                                     referent_attention.log(),
                                                                                                     gold_dist))
                                print("markable start: {}\tmarkable end: {}".format(markable_start, markable_end))
                            else:
                                word_attn_losses.append(attn_loss)

            if this_feed_ctx_attn_prob is not None and self.args.feed_attention_supervised:
                tdm_sum = this_dots_mentioned.sum(-1)
                mask = (tdm_sum > 0)
                filtered_attention = this_feed_ctx_attn_prob[mask]
                filtered_target = this_dots_mentioned[mask].float()
                filtered_target /= filtered_target.sum(-1, keepdims=True)
                if self.args.attention_supervision_method == 'kl':
                    feed_attn_loss = torch.nn.functional.kl_div(
                        filtered_attention.log(),
                        filtered_target,
                        reduction='batchmean',
                    )
                elif self.args.attention_supervision_method == 'penalize_unmentioned':
                    # feed_attn_loss = filtered_attention[filtered_target == 0].log().sum()
                    feed_attn_loss = filtered_attention[filtered_target == 0].sum()
                else:
                    raise ValueError(
                        "invalid --attention_supervision_method {}".format(self.args.attention_supervision_method))
                if feed_attn_loss != feed_attn_loss:
                    # nan
                    print("feed nan loss: {}\nthis_feed_ctx_attn_prob: {} \t this_dots_mentioned: {}".format(
                        feed_attn_loss.item(), this_feed_ctx_attn_prob.log(), this_dots_mentioned
                    ))
                else:
                    feed_attn_losses.append(feed_attn_loss)

        if ref_stats['num_dots'] == 0 or (not ref_losses):
            # not sure why I had this assert, it seems it can be tripped if you have a batch with no self mentions
            # assert ref_stats['num_dots'] == 0 and (not ref_losses)
            ref_loss = 0
        else:
            ref_loss = sum(ref_losses) / ref_stats['num_dots']

        if partner_ref_stats['num_dots'] == 0 or (not partner_ref_losses):
            # not sure why I had this assert, it seems it can be tripped if you have a batch with no *partner* mentions
            # assert partner_ref_stats['num_dots'] == 0 and (not partner_ref_losses)
            partner_ref_loss = None
        else:
            partner_ref_loss = sum(partner_ref_losses) / partner_ref_stats['num_dots']

        if (not self.args.relation_swap_augmentation) or rs_ref_stats['num_dots'] == 0 or (not relation_swapped_ref_losses):
            # not sure why I had this assert, it seems it can be tripped if you have a batch with no self mentions
            # assert ref_stats['num_dots'] == 0 and (not ref_losses)
            relation_swapped_ref_loss = 0
        else:
            relation_swapped_ref_loss = sum(relation_swapped_ref_losses) / rs_ref_stats['num_dots']

        if (not self.args.relation_swap_augmentation) or rs_partner_ref_stats['num_dots'] == 0 or (not relation_swapped_partner_ref_losses):
            # not sure why I had this assert, it seems it can be tripped if you have a batch with no self mentions
            # assert ref_stats['num_dots'] == 0 and (not ref_losses)
            relation_swapped_partner_ref_loss = 0
        else:
            relation_swapped_partner_ref_loss = sum(relation_swapped_partner_ref_losses) / rs_partner_ref_stats['num_dots']

        # print('sel_out.size(): {}'.format(sel_out.size()))
        # print('sel_tgt.size(): {}'.format(sel_tgt.size()))

        # sel_out contains the output of AttentionLayer: marginal_logits and full_log_probs, but since this is an AttentionLayer, full_log_probs is None
        # sel_out[1] should be None because we're not using a structured attention layer
        sel_logits, _, _ = sel_out

        if (~non_lang_instance_mask).any():
            sel_loss = self.sel_crit(sel_logits[~non_lang_instance_mask], sel_tgt[~non_lang_instance_mask])
            sel_correct = (sel_logits.max(dim=1)[1] == sel_tgt).sum().item()
            sel_total = sel_logits.size(0)
        else:
            sel_loss = torch.tensor(0.0, requires_grad=True)
            sel_correct = 0.0
            sel_total = 0.0

        if word_attn_losses:
            word_attn_loss = sum(word_attn_losses) / len(word_attn_losses)
        else:
            word_attn_loss = None

        if feed_attn_losses:
            feed_attn_loss = sum(feed_attn_losses) / len(feed_attn_losses)
        else:
            feed_attn_loss = None

        next_mention_losses = []
        next_mention_stop_losses = []
        next_mention_stats = defaultdict(lambda: 0.0)
        # next_mention_stats = {'num_dots': 0}

        expanded_next_mention_loss = self.args.next_mention_prediction_type == 'multi_reference'

        if self.args.next_mention_prediction:
            assert len(dots_mentioned) + 1 == len(next_mention_outs)
            for i in range(len(dots_mentioned)):

                pred_dots_mentioned, next_mention_stop_loss, next_mention_num_markables = next_mention_outs[i]
                if self.args.next_mention_prediction_type == 'multi_reference':
                    gold_num_mentions = dots_mentioned_num_markables[i].long()
                    gold_dots_mentioned = dots_mentioned_per_ref[i].long()
                    if (~non_lang_instance_mask).any():
                        next_mention_stop_losses.append(next_mention_stop_loss[~non_lang_instance_mask].sum())
                elif self.args.next_mention_prediction_type == 'collapsed':
                    # supervise only for self, so pseudo_num_mentions = 1 iff is_self; 0 otherwise
                    gold_num_mentions = is_self[i].long()
                    gold_dots_mentioned = dots_mentioned[i].long().unsqueeze(1)
                else:
                    raise ValueError(f"--next_mention_prediction_type={self.args.next_mention_prediction}")

                if pred_dots_mentioned is not None:
                    # TODO: fix this, it should check if gold dots is not empty
                    # hack; pass True for inpt because this method only uses it to ensure it's not null
                    if (~non_lang_instance_mask).any():
                        _loss, _pred, _stats = self._ref_loss(
                            True, gold_dots_mentioned, pred_dots_mentioned, gold_num_mentions,
                            # TODO: collapse param naming is misleading; collapse adds in additional expanded_* terms
                            collapse=expanded_next_mention_loss,
                            mask=~non_lang_instance_mask
                        )
                        next_mention_stats = utils.sum_dicts(next_mention_stats, _stats)
                        # print("i: {}\tgold_dots_mentioned.sum(): {}\t(pred_dots_mentioned > 0).sum(): {}".format(i, gold_dots_mentioned.sum(), (pred_dots_mentioned > 0).sum()))
                        # print("{} / {}".format(_correct, _total))
                        next_mention_losses.append(_loss)

        is_selection_losses = []
        is_selection_stats = defaultdict(lambda: 0.0)

        if self.args.is_selection_prediction and (~non_lang_instance_mask).any():
            for is_sel, is_sel_out in utils.safe_zip(is_selection,is_selection_outs):
                is_sel_loss = self.is_sel_crit(is_sel_out[~non_lang_instance_mask], is_sel[~non_lang_instance_mask].float())
                is_selection_losses.append(is_sel_loss)

                add_selection_stats(is_sel, is_sel_out, is_selection_stats)

        if is_selection_losses:
            is_selection_loss = sum(is_selection_losses) / len(is_selection_losses)
        else:
            is_selection_loss = None

        if next_mention_stats['num_dots'] == 0 or (not next_mention_losses):
            # not sure why I had this assert, it seems it can be tripped if you have a batch with no *next* mentions
            # assert next_mention_stats['num_dots'] == 0 and (not next_mention_losses)
            next_mention_loss = None
        else:
            loss_normalizer_name = 'expanded_num_dots' if expanded_next_mention_loss else 'num_dots'
            assert next_mention_stats[loss_normalizer_name] != 0
            next_mention_loss = sum(next_mention_losses) / next_mention_stats[loss_normalizer_name]

        if not next_mention_stop_losses:
            next_mention_stop_loss = None
        else:
            loss_normalizer_name = 'expanded_exact_match_denom' if expanded_next_mention_loss else 'exact_match_denom'
            loss_normalizer = next_mention_stats[loss_normalizer_name]
            if loss_normalizer == 0:
                next_mention_stop_loss = None
            else:
                next_mention_stop_loss = sum(next_mention_stop_losses) / next_mention_stats[loss_normalizer_name]

        return ForwardRet(
            lang_loss=lang_loss,
            spatial_lang_loss=spatial_lang_loss,
            unnormalized_lang_loss=unnormed_lang_loss.item(),
            num_words=total_lens_no_eos,
            unnormalized_spatial_lang_loss=unnormed_spatial_lang_loss.item(),
            num_spatial_words=num_spatial_words,
            ref_loss=ref_loss,
            sel_loss=sel_loss,
            word_attn_loss=word_attn_loss,
            feed_attn_loss=feed_attn_loss,
            sel_correct=sel_correct,
            sel_num_dots=sel_total,
            ref_stats=ref_stats,
            partner_ref_loss=partner_ref_loss,
            partner_ref_stats=partner_ref_stats,
            next_mention_loss=next_mention_loss,
            next_mention_stop_loss=next_mention_stop_loss,
            next_mention_stats=next_mention_stats,
            is_selection_loss=is_selection_loss,
            is_selection_stats=is_selection_stats,
            l1_loss=l1_loss,
            relation_swapped_ref_loss=relation_swapped_ref_loss,
            relation_swapped_partner_ref_loss=relation_swapped_partner_ref_loss,
        )
