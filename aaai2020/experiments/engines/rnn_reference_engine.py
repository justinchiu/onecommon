import string
from collections import namedtuple
import copy
import pprint
import time

import math
import numpy as np
import pyro
import torch
import tqdm
from torch import nn
from torch.autograd import Variable

from collections import defaultdict

import utils
from engines import EngineBase

from engines.beliefs import BeliefConstructor

ForwardRet = namedtuple(
    "ForwardRet",
    ['lang_loss', 'unnormalized_lang_loss',
     'word_attn_loss', 'feed_attn_loss',
     'num_words',
     'ref_loss', 'ref_stats',
     'sel_loss', 'sel_correct', 'sel_num_dots',
     # 'ref_gold_positive', 'ref_pred_positive', 'ref_true_positive', 'ref_num_dots', 'ref_em_num', 'ref_em_denom',
     # 'attn_ref_stats',
     'partner_ref_loss', 'partner_ref_stats',
     # 'partner_ref_correct', 'partner_ref_gold_positive', 'partner_ref_pred_positive', 'partner_ref_true_positive', 'partner_ref_num_dots', 'partner_ref_em_num', 'partner_ref_em_denom',
     'next_mention_loss', 'next_mention_stats',
     # 'next_mention_correct', 'next_mention_gold_positive', 'next_mention_pred_positive',
     # 'next_mention_true_positive', 'next_mention_num_dots', 'next_mention_em_num', 'next_mention_em_denom',
     ],
)


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

def add_metrics(metric_dict_src, metric_dict_tgt, prefix):
    gold_positive = metric_dict_src['{}_gold_positive'.format(prefix)]
    pred_positive = metric_dict_src['{}_pred_positive'.format(prefix)]
    true_positive = metric_dict_src['{}_true_positive'.format(prefix)]
    correct = metric_dict_src['{}_correct'.format(prefix)]
    num_dots = metric_dict_src['{}_num_dots'.format(prefix)]

    em_num = metric_dict_src['{}_exact_match_num'.format(prefix)]
    em_denom = metric_dict_src['{}_exact_match_denom'.format(prefix)]

    precision = true_positive / pred_positive if pred_positive > 0 else 0
    recall = true_positive / gold_positive if gold_positive > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metric_dict_tgt['{}_accuracy'.format(prefix)] = correct / num_dots if num_dots > 0 else 0
    metric_dict_tgt['{}_precision'.format(prefix)] = precision
    metric_dict_tgt['{}_recall'.format(prefix)] = recall
    metric_dict_tgt['{}_f1'.format(prefix)] = f1

    metric_dict_tgt['{}_exact_match'.format(prefix)] = em_num / em_denom if em_denom > 0 else 0

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

    def _forward(self, batch):
        assert not self.args.word_attention_supervised, 'this only makes sense for a hierarchical model, and --lang_only_self'
        assert not self.args.feed_attention_supervised, 'this only makes sense for a hierarchical model, and --lang_only_self'
        assert not self.args.mark_dots_mentioned, 'this only makes sense for a hierarchical model, and --lang_only_self'
        ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, _, _, _, _, sel_idx, lens, partner_ref_inpt, partner_ref_tgt_our_view, partner_num_markables = batch

        ctx = Variable(ctx)
        inpt = Variable(inpt)
        bsz = ctx.size(0)
        if ref_inpt is not None:
            ref_inpt = Variable(ref_inpt)

        if ref_tgt is not None:
            dots_mentioned = make_dots_mentioned(ref_tgt, self.args)
        else:
            dots_mentioned = None

        out, (ref_out, partner_ref_out), sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_out, lang_h = \
            self.model.forward(
                ctx, inpt, ref_inpt, sel_idx,
                num_markables=None, # todo: fix this to be a vector of (bsz,) with constant value determined by the size of ref_tgt
                partner_num_markables=partner_num_markables,
                lens=None, dots_mentioned=dots_mentioned,
                belief_constructor=None, partner_ref_inpt=partner_ref_inpt,
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
            next_mention_stats={},
        )

    def train_batch(self, batch, epoch):
        forward_ret = self._forward(batch)

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
        if self.args.sel_weight > 0:
            loss += self.args.sel_weight * forward_ret.sel_loss
        if self.args.ref_weight > 0 and forward_ret.ref_loss is not None:
            loss += self.args.ref_weight * forward_ret.ref_loss
        if self.args.partner_ref_weight > 0 and forward_ret.partner_ref_loss is not None:
            loss += self.args.partner_ref_weight * forward_ret.partner_ref_loss
        if self.args.next_mention_weight > 0 and forward_ret.next_mention_loss is not None:
            if not self.args.next_mention_start_epoch or (epoch >= self.args.next_mention_start_epoch):
                loss += self.args.next_mention_weight * forward_ret.next_mention_loss

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

    def valid_batch(self, batch, epoch):
        with torch.no_grad():
            return self._forward(batch)

    def test_batch(self, batch, epoch):
        with torch.no_grad():
            return self._forward(batch)

    def _pass(self, dataset, batch_fn, split_name, use_tqdm, epoch):
        start_time = time.time()

        metrics = {}

        for batch in tqdm.tqdm(dataset, ncols=80) if use_tqdm else dataset:
            # for batch in trainset:
            # lang_loss, ref_loss, ref_correct, ref_total, sel_loss, word_attn_loss, feed_attn_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = batch_fn(batch)
            forward_ret = batch_fn(batch, epoch)
            batch_metrics = flatten_metrics(forward_ret._asdict())
            metrics = utils.sum_dicts(metrics, batch_metrics)

        print("{} word_attn_loss: {:.4f}".format(split_name, metrics['word_attn_loss']))
        print("{} feed_attn_loss: {:.4f}".format(split_name, metrics['feed_attn_loss']))

        # pprint.pprint({'{}_{}'.format(name, k): v for k, v in total_attn_ref_stats.items()})

        time_elapsed = time.time() - start_time

        aggregate_metrics = {
            'lang_loss': metrics['lang_loss'] / len(dataset),
            'ref_loss': metrics['ref_loss'] / len(dataset),
            'partner_ref_loss': metrics['partner_ref_loss'] / len(dataset),
            'next_mention_loss': metrics['next_mention_loss'] / len(dataset),
            'select_loss': metrics['sel_loss'] / len(dataset),
            # do select_accuracy here b/c we won't call add_metrics on select
            'select_accuracy': metrics['sel_correct'] / metrics['sel_num_dots'],
            'time': time_elapsed,
            'correct_ppl': np.exp(metrics['unnormalized_lang_loss'] / metrics['num_words']),
        }
        add_metrics(metrics, aggregate_metrics, "ref")
        add_metrics(metrics, aggregate_metrics, "partner_ref")
        add_metrics(metrics, aggregate_metrics, "next_mention")
        return aggregate_metrics

    def train_pass(self, trainset, trainset_stats, epoch):
        '''
        basic implementation of one training pass
        '''
        self.model.train()
        return self._pass(trainset, self.train_batch, "train", use_tqdm=True, epoch=epoch)

    def valid_pass(self, validset, validset_stats, epoch):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()
        return self._pass(validset, self.valid_batch, "val", use_tqdm=False, epoch=epoch)

    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_metrics = self.train_pass(trainset, trainset_stats, epoch)
        valid_metrics = self.valid_pass(validset, validset_stats, epoch)

        if self.verbose:
            print('epoch %03d \t s/epoch %.2f \t lr %.2E' % (epoch, train_metrics['time'], lr))

            for split_name, metrics in [('train', train_metrics), ('valid', valid_metrics)]:
                metrics = metrics.copy()
                metrics['ppl(avg exp lang_loss)'] = np.exp(metrics['lang_loss'])
                metrics['lang_loss'] *= self.args.lang_weight
                metrics['select_loss'] *= self.args.sel_weight
                metrics['ref_loss'] *= self.args.ref_weight
                metrics['partner_ref_loss'] *= self.args.partner_ref_weight
                metrics['next_mention_loss'] *= self.args.next_mention_weight

                quantities = [
                    ['lang_loss', 'ppl(avg exp lang_loss)', 'correct_ppl'],
                    ['select_loss', 'select_accuracy'],
                    ['ref_loss', 'ref_accuracy', 'ref_precision', 'ref_recall', 'ref_f1', 'ref_exact_match'],
                    ['partner_ref_loss', 'partner_ref_accuracy', 'partner_ref_precision', 'partner_ref_recall',
                     'partner_ref_f1', 'partner_ref_exact_match'],
                    ['next_mention_loss', 'next_mention_accuracy', 'next_mention_precision', 'next_mention_recall',
                     'next_mention_f1', 'next_mention_exact_match'],
                ]
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

        # TODO: add next mention loss
        return metrics['lang_loss'] * self.args.lang_weight \
               + metrics['select_loss'] * self.args.sel_weight \
               + metrics['ref_loss'] * self.args.ref_weight \
               + metrics['partner_ref_loss'] * self.args.partner_ref_weight \
               + metrics['next_mention_loss'] * self.args.next_mention_weight

    def train(self, corpus, model_filename_fn):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        validdata = corpus.valid_dataset(self.args.bsz)

        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            print("train set stats:")
            pprint.pprint(traindata[1])

            metrics = self.iter(epoch, self.opt.param_groups[0]["lr"], traindata, validdata)

            # valid_lang_loss, valid_select_loss, valid_reference_loss, valid_select_acc = self.iter(
            # )

            combined_valid_loss = self.combine_loss(metrics['valid'])

            if self.scheduler is not None:
                self.scheduler.step(combined_valid_loss)

            if combined_valid_loss < best_combined_valid_loss:
                print(
                    f"update best model -- valid combined: {combined_valid_loss:.4f}\t" +
                    '\t'.join(f'{name} {metrics["valid"][name]:.4f}' for name in ['lang_loss', 'select_loss', 'select_accuracy', 'ref_loss', 'partner_ref_loss', 'next_mention_loss'])
                )
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

                # utils.save_model(best_model, model_filename_fn('ep-{}'.format(epoch), 'th'), prefix_dir=None)
                # utils.save_model(best_model.state_dict(), model_filename_fn('ep-{}'.format(epoch), 'stdict'), prefix_dir=None)

        return best_combined_valid_loss, best_model


def bit_to_int_array(bit_array, base=2):
    int_array = torch.zeros(bit_array.size()[:-1]).to(bit_array.device).long()
    N = bit_array.size(-1)
    for ix in range(N):
        int_array *= base
        int_array += bit_array[..., ix]
    return int_array


def int_to_bit_array(int_array, num_bits=None, base=2):
    assert int_array.dtype in [torch.int16, torch.int32, torch.int64]
    assert (int_array >= 0).all()
    if num_bits is None:
        num_bits = (int_array.max().float().log() / math.log(base)).floor().long().item() + 1
    int_array_flat = int_array.view(-1)
    N = int_array_flat.size(0)
    ix = torch.arange(N)
    bits = torch.zeros(N, num_bits).to(int_array.device).long()
    remainder = int_array_flat
    for i in range(num_bits):
        bits[ix, num_bits - i - 1] = remainder % base
        remainder = remainder / base
    assert (remainder == 0).all()
    bits = bits.view((int_array.size()) + (num_bits,))
    return bits



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

    def compute_stats(self, ref_mask, ref_tgt, ref_tgt_ix=None, ref_pred=None, ref_pred_ix=None):
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
            'correct': ((ref_pred == ref_tgt.long()) * ref_mask.byte()).sum().item(),
            'num_dots': ref_mask.sum().item(),
            'gold_positive': ref_tgt.sum().item(),
            'pred_positive': (ref_pred * ref_mask.byte()).sum().item(),
            'true_positive': (ref_pred & ref_tgt.bool()).sum().item(),
            'exact_match_num': ref_exact_matches.sum().float().item(),
            'exact_match_denom': ref_mask_instance_level.sum().float().item(),
        }

        assert stats['pred_positive'] >= stats['true_positive']
        assert stats['gold_positive'] >= stats['true_positive']
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
        ref_pred[:,has_multiple] = int_to_bit_array(ref_pred_ix[:,has_multiple], num_bits=num_dots)

        gold_parts = LinearChainNoScanCRF.struct.to_parts(
            ref_tgt_ix_transpose, 2**num_dots, lengths=num_markables
        )
        if self.args.structured_temporal_attention_training == 'likelihood':
            log_likelihoods = temporal_dist.log_prob(gold_parts)
            losses = -log_likelihoods
        elif self.args.structured_temporal_attention_training == 'max_margin':
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

    def forward(self, ref_inpt, ref_tgt, ref_out, num_markables):
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

        stats = self.compute_stats(ref_mask, ref_tgt, ref_tgt_ix, ref_pred, ref_pred_ix)

        return ref_loss, ref_pred, stats


class PragmaticReferencePredictor(ReferencePredictor):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--l1_sample', action='store_true', help='l1 listener should sample (otherwise top-k)')
        parser.add_argument('--l1_candidates', type=int, default=10, help='number of dot configurations for l1 to consider')
        parser.add_argument('--l1_speaker_weight', type=float, default=1.0, help='(1 - lambda) * l0_log_prob + (lambda) * s0_log_prob')

    def __init__(self, args):
        super().__init__(args)
        self._logit_to_full_einsum_str = None

    def logit_to_full_einsum_str(self, logits):
        if self._logit_to_full_einsum_str is not None:
            return self._logit_to_full_einsum_str

        mentions, bsz, num_ent = logits.size()
        var_names = string.ascii_lowercase[:num_ent]
        batch_name = 'z'
        mention_name = 'y'
        assert batch_name not in var_names
        assert mention_name not in var_names

        self._logit_to_full_einsum_str = '{}->{}'.format(
            ','.join('{}{}{}'.format(mention_name, batch_name, var_name) for var_name in var_names),
            '{}{}{}'.format(mention_name, batch_name, ''.join(var_names))
        )
        return self._logit_to_full_einsum_str

    def logits_to_full(self, logits):
        num_ent = logits.size(-1)
        einsum_str = self.logit_to_full_einsum_str(logits)

        # mentions x bsz x num_ent x 2
        stack_logits = torch.stack((-logits, logits), dim=-1)
        assert stack_logits.dim() == 4

        # num_ent arrays, each of size mentions x bsz x 2
        factored_logits = (l.squeeze(-2) for l in stack_logits.split(1, dim=-2))

        outputs = pyro.ops.contract.einsum(
            einsum_str,
            *factored_logits,
            modulo_total=True,
            backend='pyro.ops.einsum.torch_log',
        )
        assert len(outputs) == 1
        output = outputs[0]
        assert output.dim() == 2 + num_ent
        return output

    def forward(self, ref_inpt, ref_tgt, ref_out, num_markables, scoring_function):
        k = self.args.l1_candidates
        if ref_inpt is None or ref_out is None:
            return None, None, self.empty_stats()

        ref_tgt_p, ref_mask = self.preprocess(ref_tgt, num_markables)

        ref_out_logits, ref_out_full, ref_dist = ref_out

        N, bsz, num_dots = ref_out_logits.size()

        candidate_l0_scores = torch.zeros(N, bsz, k).to(ref_inpt.device)
        candidate_indices = torch.zeros(N, bsz, k).long().to(ref_inpt.device)

        if ref_dist is not None:
            assert not self.args.l1_sample

            use_temporal = num_markables > 1

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
            # tile across mention dimension
            # N x bsz x k
            candidate_l0_scores_multi = candidate_l0_scores_multi.unsqueeze(0).repeat_interleave(N, dim=0)
            candidate_l0_scores[:,use_temporal] = candidate_l0_scores_multi[:,use_temporal]

        else:
            use_temporal = torch.zeros_like(num_markables).bool()

        ## get scores and candidates for batch items where use_temporal == False
        if ref_out_full is None:
            ref_out_full = self.logits_to_full(ref_out_logits).contiguous()

        num_ent = ref_tgt_p.size(-1)
        # num_mentions x bsz x
        assert ref_out_full.dim() == 2 + num_ent

        ref_out_full_reshape = ref_out_full.view(N, bsz, 2**num_ent)
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

        ## now that candidate_indices and candidate_l0_scores are created, rescore them

        # (N*bsz) x k x 7
        candidate_dots = int_to_bit_array(candidate_indices, num_bits=num_ent)
        candidate_dots = candidate_dots.view(N, bsz, k, num_ent)

        # N x bsz x k
        candidate_s0_scores = scoring_function(candidate_dots)
        assert candidate_s0_scores.size() == candidate_l0_scores.size()

        lmbd = self.args.l1_speaker_weight
        # N x bsz x k
        joint_scores = (1 - lmbd) * candidate_l0_scores + lmbd * candidate_s0_scores

        # N x bsz with values [0..k-1]
        chosen = joint_scores.argmax(-1)

        chosen_indices = candidate_indices.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

        # convert indices to bits
        ref_pred = int_to_bit_array(chosen_indices, num_bits=num_ent)

        if self.args.l1_candidates == 1 and self.args.l1_speaker_weight == 0.0:
            ref_pred_l0 = super().forward(ref_inpt, ref_tgt, ref_out, num_markables)[1]
            assert (ref_pred == ref_pred_l0).all()

        stats = self.compute_stats(ref_mask, ref_tgt_p, ref_pred=ref_pred)
        ref_loss = 0.0
        return ref_loss, ref_pred, stats


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

    def _ref_loss(self, *args):
        return self.ref_loss.forward(*args)

    def _forward(self, batch):
        if self.args.word_attention_supervised or self.args.feed_attention_supervised or self.args.mark_dots_mentioned:
            assert self.args.lang_only_self
        ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, scenario_ids, real_ids, partner_real_ids, _, _, sel_idx, lens, rev_idxs, hid_idxs, num_markables, is_self, partner_ref_inpts, partner_ref_tgts_our_view, all_partner_num_markables, ref_disagreements, partner_ref_disagreements = batch

        ctx = Variable(ctx)
        bsz = ctx.size(0)
        num_dots = int(ctx.size(1) / 4)
        assert num_dots == 7

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

        dots_mentioned = make_dots_mentioned_multi(ref_tgts, self.args, bsz, num_dots)
        partner_dots_mentioned_our_view = make_dots_mentioned_multi(partner_ref_tgts_our_view, self.args, bsz, num_dots)

        belief_constructor = BeliefConstructor(
            self.args, bsz, num_dots, inpts, ref_tgts, partner_ref_tgts_our_view,
            real_ids, partner_real_ids, sel_tgt, is_self, partner_dots_mentioned_our_view, dots_mentioned
        )

        outs, ref_outs_and_partner_ref_outs, sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_outs, lang_h, ctx_h, ctx_differences = self.model.forward(
            ctx, inpts, ref_inpts, sel_idx,
            num_markables, all_partner_num_markables,
            lens, dots_mentioned,
            belief_constructor=belief_constructor,
            partner_ref_inpts=partner_ref_inpts,
        )

        sel_tgt = Variable(sel_tgt)
        lang_losses = []
        assert len(inpts) == len(tgts) == len(outs) == len(lens)
        for i, (out, tgt) in enumerate(zip(outs, tgts)):
            # T x bsz
            loss = self.crit_no_reduce(out, tgt).view(-1, bsz)
            if self.args.lang_only_self:
                # loss = loss * (this_is_self.unsqueeze(0).expand_as(loss))
                loss = loss * (is_self[i].unsqueeze(0).expand_as(loss))
            lang_losses.append(loss.sum())
        total_lens = sum(l.sum() for l in lens)

        # w1 w2 w3 ... <eos> YOU:
        # or
        # w1 w2 w3 ... <eos> THEM:
        # subtract 2 to remove <eos> and YOU/THEM:
        total_lens_no_eos = sum((l - 2).sum().item() for l in lens)

        unnormed_lang_loss = sum(lang_losses)
        lang_loss = unnormed_lang_loss / total_lens

        ref_losses = []
        # other keys and values will be added
        # ref_stats = {'num_dots': 0}
        ref_stats = defaultdict(lambda: 0.0)

        partner_ref_losses = []
        # other keys and values will be added
        # partner_ref_stats = {'num_dots': 0}
        partner_ref_stats = defaultdict(lambda: 0.0)

        attn_ref_true_positive = 0
        attn_ref_total = 0
        attn_ref_gold_positive = 0
        attn_ref_pred_positive = 0

        word_attn_losses = []
        feed_attn_losses = []

        assert len(ref_inpts) == len(ref_tgts) == len(num_markables)

        assert len(partner_ref_inpts) == len(partner_ref_tgts_our_view) == len(all_partner_num_markables)

        # TODO: just index into the lists; the safety check isn't worth it
        for ref_inpt, partner_ref_inpt, (ref_out, partner_ref_out), ref_tgt, partner_ref_tgt,\
            this_num_markables,this_partner_num_markables, this_ctx_attn_prob, this_feed_ctx_attn_prob, \
            this_dots_mentioned, inpt, tgt in utils.safe_zip(
            ref_inpts, partner_ref_inpts, ref_outs_and_partner_ref_outs, ref_tgts, partner_ref_tgts_our_view,
            num_markables, all_partner_num_markables, ctx_attn_prob, feed_ctx_attn_prob,
            dots_mentioned, inpts, tgts
        ):
            if (this_num_markables == 0).all() or ref_tgt is None:
                continue
            assert max(this_num_markables) == ref_tgt.size(1)
            _ref_loss, _ref_pred, _ref_stats = self._ref_loss(
                ref_inpt, ref_tgt, ref_out, this_num_markables
            )
            ref_losses.append(_ref_loss)
            ref_stats = utils.sum_dicts(ref_stats, _ref_stats)

            _partner_ref_loss, _partner_ref_pred, _partner_ref_stats = self._ref_loss(
                partner_ref_inpt, partner_ref_tgt, partner_ref_out, this_partner_num_markables
            )
            partner_ref_stats = utils.sum_dicts(partner_ref_stats, _partner_ref_stats)
            if _partner_ref_loss is not None:
                partner_ref_losses.append(_partner_ref_loss)

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
        ref_loss = sum(ref_losses) / ref_stats['num_dots']

        if partner_ref_stats['num_dots'] == 0 or (not partner_ref_losses):
            assert partner_ref_stats['num_dots'] == 0 and (not partner_ref_losses)
            partner_ref_loss = None
        else:
            partner_ref_loss = sum(partner_ref_losses) / partner_ref_stats['num_dots']

        # print('sel_out.size(): {}'.format(sel_out.size()))
        # print('sel_tgt.size(): {}'.format(sel_tgt.size()))

        # sel_out contains the output of AttentionLayer: marginal_logits and full_log_probs, but since this is an AttentionLayer, full_log_probs is None
        # sel_out[1] should be None because we're not using a structured attention layer
        sel_logits, _, _ = sel_out

        sel_loss = self.sel_crit(sel_logits, sel_tgt)
        sel_correct = (sel_logits.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_logits.size(0)

        if word_attn_losses:
            word_attn_loss = sum(word_attn_losses) / len(word_attn_losses)
        else:
            word_attn_loss = None

        if feed_attn_losses:
            feed_attn_loss = sum(feed_attn_losses) / len(feed_attn_losses)
        else:
            feed_attn_loss = None

        next_mention_losses = []
        next_mention_stats = defaultdict(lambda: 0.0)
        # next_mention_stats = {'num_dots': 0}

        if self.args.next_mention_prediction:
            assert len(dots_mentioned) + 1 == len(next_mention_outs)
            for i in range(len(dots_mentioned)):
                # supervise only for self, so pseudo_num_mentions = 1 iff is_self; 0 otherwise
                pseudo_num_mentions = is_self[i].long()
                gold_dots_mentioned = dots_mentioned[i].long().unsqueeze(1)
                pred_dots_mentioned_logits = next_mention_outs[i]

                # hack; pass True for inpt because this method only uses it to ensure it's not null
                _loss, _pred, _stats = self._ref_loss(
                    True, gold_dots_mentioned, pred_dots_mentioned_logits, pseudo_num_mentions
                )
                next_mention_stats = utils.sum_dicts(next_mention_stats, _stats)
                # print("i: {}\tgold_dots_mentioned.sum(): {}\t(pred_dots_mentioned > 0).sum(): {}".format(i, gold_dots_mentioned.sum(), (pred_dots_mentioned > 0).sum()))
                # print("{} / {}".format(_correct, _total))
                next_mention_losses.append(_loss)

        if next_mention_stats['num_dots'] == 0 or (not next_mention_losses):
            assert next_mention_stats['num_dots'] == 0 and (not next_mention_losses)
            next_mention_loss = None
        else:
            next_mention_loss = sum(next_mention_losses) / next_mention_stats['num_dots']

        return ForwardRet(
            lang_loss=lang_loss,
            unnormalized_lang_loss=unnormed_lang_loss.item(),
            num_words=total_lens_no_eos,
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
            next_mention_stats=next_mention_stats,
        )
