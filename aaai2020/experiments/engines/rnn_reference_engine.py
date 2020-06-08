from collections import namedtuple
import copy
import pprint
import time

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

import utils
from engines import EngineBase

ForwardRet = namedtuple(
    "ForwardRet",
    ['lang_loss', 'ref_loss', 'ref_correct', 'sel_loss',
     'word_attn_loss', 'feed_attn_loss',
     'sel_correct', 'sel_total',
     'ref_positive', 'ref_total',
     'attn_ref_stats',
     'partner_ref_loss', 'partner_ref_correct', 'partner_ref_positive', 'partner_ref_total',
     'next_mention_loss', 'next_mention_correct', 'next_mention_positive', 'next_mention_total',
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
        return ref_tgt[:,0,:] > 0
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


def _make_beliefs(
    bsz, num_dots, args, inpts, ref_tgts, partner_ref_tgts_our_view, real_ids, partner_real_ids, sel_tgt, is_self,
    partner_ref_outs,
    timestep,
    partner_dots_mentioned_our_view, # one per sentence
    dots_mentioned, # one per sentence
):

    def make_beliefs(beliefs_list, arg_name):
        all_beliefs = []
        for beliefs_name in beliefs_list:
            if beliefs_name == 'selected':
                # bsz x num_dots x 1, one-hot if that dot is the one selected
                beliefs = torch.zeros(sel_tgt.size(0), num_dots).scatter(1, sel_tgt.unsqueeze(1), 1).unsqueeze(-1)
            elif beliefs_name == 'partners':
                partner_has = []
                for batch_ix, (a_rids, p_rids) in enumerate(utils.safe_zip(real_ids, partner_real_ids)):
                    p_rids = set(p_rids)
                    partner_has.append([a_rid in p_rids for a_rid in a_rids])
                beliefs = torch.tensor(partner_has).float().unsqueeze(-1)
            elif beliefs_name in ['last_partner_mentioned', 'last_partner_mentioned_predicted']:
                if beliefs_name == 'last_partner_mentioned':
                    mentions = partner_dots_mentioned_our_view
                else:
                    mentions = [
                        out.sigmoid().max(0).values if out is not None else None
                        for out in partner_ref_outs
                    ]
                    if args.detach_beliefs:
                        mentions = [
                            mention.detach() if mention is not None else None
                            for mention in mentions
                        ]
                    mentions = [
                        mention if mention is not None else torch.zeros(bsz, num_dots)
                        for mention in mentions
                    ]
                beliefs = torch.zeros(bsz, num_dots)
                if timestep > 0:
                    ts = (torch.tensor([timestep] * bsz).long() - 1) - is_self[timestep-1].long()
                    for j, t_j in enumerate(ts):
                        if t_j >= 0:
                            beliefs[j] = mentions[t_j][j]
                beliefs = beliefs.float().unsqueeze(-1)
            elif beliefs_name == 'this_partner_mentioned':
                if timestep >= 0:
                    beliefs = partner_dots_mentioned_our_view[timestep].float().unsqueeze(-1)
                else:
                    beliefs = torch.zeros_like(partner_dots_mentioned_our_view[0]).float().unsqueeze(-1)
            elif beliefs_name == 'cumulative_partner_mentioned':
                beliefs = torch.zeros(bsz, num_dots).bool()
                if timestep >= 0:
                    for t in range(timestep):
                        beliefs |= partner_dots_mentioned_our_view[t]
                beliefs = beliefs.float().unsqueeze(-1)
            elif beliefs_name == 'this_mentioned':
                if timestep >= 0:
                    beliefs = dots_mentioned[timestep].float().unsqueeze(-1)
                else:
                    beliefs = torch.zeros_like(dots_mentioned[0]).float().unsqueeze(-1)
            elif beliefs_name == 'next_mentioned':
                if timestep < len(dots_mentioned) - 1:
                    beliefs = dots_mentioned[timestep+1].float().unsqueeze(-1)
                else:
                    beliefs = torch.zeros_like(dots_mentioned[0]).float().unsqueeze(-1)
            else:
                raise ValueError('invalid --{} {}'.format(arg_name, beliefs_name))
            all_beliefs.append(beliefs)
        if all_beliefs:
            if len(all_beliefs) > 1:
                return torch.cat(all_beliefs, dim=-1)
            else:
                return all_beliefs[0]
        else:
            return None

    if hasattr(args, 'selection_beliefs'):
        if timestep == len(inpts) - 1:
            selection_beliefs = make_beliefs(args.selection_beliefs, 'selection_beliefs')
        else:
            selection_beliefs = None
    else:
        selection_beliefs = None
    if hasattr(args, 'generation_beliefs'):
        generation_beliefs = make_beliefs(args.generation_beliefs, 'generation_beliefs')
    else:
        generation_beliefs = None

    if hasattr(args, 'mention_beliefs'):
        mention_beliefs = make_beliefs(args.mention_beliefs, 'mention_beliefs')
    else:
        mention_beliefs = None

    return selection_beliefs, generation_beliefs, mention_beliefs

class RnnReferenceEngine(EngineBase):
    @classmethod
    def add_args(cls, parser):
        pass
    def __init__(self, model, args, verbose=False):
        super(RnnReferenceEngine, self).__init__(model, args, verbose)

    def _ref_loss(self, ref_inpt, ref_tgt, ref_out):
        if ref_inpt is not None:
            ref_tgt = Variable(ref_tgt)
            ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
            ref_loss = self.ref_crit(ref_out, ref_tgt)
            ref_correct = ((ref_out > 0).long() == ref_tgt.long()).sum().item()
            ref_total = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)
            ref_positive = ref_tgt.sum().item()
        else:
            ref_loss = None
            ref_correct = 0
            ref_total = 0
            ref_positive = 0
        return ref_loss, ref_correct, ref_total, ref_positive

    def _forward(self, batch):
        assert not self.args.word_attention_supervised, 'this only makes sense for a hierarchical model, and --lang_only_self'
        assert not self.args.feed_attention_supervised, 'this only makes sense for a hierarchical model, and --lang_only_self'
        assert not self.args.mark_dots_mentioned, 'this only makes sense for a hierarchical model, and --lang_only_self'
        ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, _, _, _, _, sel_idx, lens, partner_ref_inpt, partner_ref_tgt_our_view, partner_num_markables = batch

        ctx = Variable(ctx)
        inpt = Variable(inpt)
        if ref_inpt is not None:
            ref_inpt = Variable(ref_inpt)

        if ref_tgt is not None:
            dots_mentioned = make_dots_mentioned(ref_tgt, self.args)
        else:
            dots_mentioned = None

        out, (ref_out, partner_ref_out), sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_out = self.model.forward(
            ctx, inpt, ref_inpt, sel_idx, lens=None, dots_mentioned=dots_mentioned,
            selection_beliefs=None, generation_beliefs=None
        )

        tgt = Variable(tgt)
        sel_tgt = Variable(sel_tgt)
        lang_loss = self.crit(out, tgt)

        ref_loss, ref_correct, ref_total, ref_positive = self._ref_loss(
            ref_inpt, ref_tgt, ref_out
        )

        partner_ref_loss, partner_ref_correct, partner_ref_total, partner_ref_positive = self._ref_loss(
            partner_ref_inpt, partner_ref_tgt_our_view, partner_ref_out
        )

        sel_loss = self.sel_crit(sel_out, sel_tgt)
        sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_out.size(0)

        # TODO
        attn_ref_stats = {}

        # TODO
        word_attn_loss = None
        feed_attn_loss = None

        return ForwardRet(
            lang_loss=lang_loss,
            ref_loss=ref_loss,
            ref_correct=ref_correct,
            sel_loss=sel_loss,
            word_attn_loss=word_attn_loss,
            feed_attn_loss=feed_attn_loss,
            sel_correct=sel_correct,
            sel_total=sel_total,
            ref_positive=ref_positive,
            ref_total=ref_total,
            attn_ref_stats=attn_ref_stats,
            partner_ref_loss=partner_ref_loss,
            partner_ref_correct=partner_ref_correct,
            partner_ref_positive=partner_ref_positive,
            partner_ref_total=partner_ref_total,
            # TODO
            next_mention_loss=None,
            next_mention_correct=0,
            next_mention_positive=0,
            next_mention_total=0,
        )


    def train_batch(self, batch):
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

        loss = 0
        if self.args.lang_weight > 0:
            loss += self.args.lang_weight * forward_ret.lang_loss
        if self.args.sel_weight > 0:
            loss += self.args.sel_weight * forward_ret.sel_loss
        if self.args.ref_weight > 0 and forward_ret.ref_loss is not None:
            loss += self.args.ref_weight * forward_ret.ref_loss
        if self.args.partner_ref_weight > 0 and forward_ret.partner_ref_loss is not None:
            loss += self.args.partner_ref_weight * forward_ret.partner_ref_loss
        if self.args.next_mention_weight > 0 and forward_ret.next_mention_loss is not None:
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

    def valid_batch(self, batch):
        with torch.no_grad():
            return self._forward(batch)

    def test_batch(self, batch):
        with torch.no_grad():
            return self._forward(batch)

    def _pass(self, dataset, batch_fn, name, use_tqdm):
        total_lang_loss, total_select_loss, total_select, total_select_correct = 0, 0, 0, 0
        start_time = time.time()

        total_ref_loss, total_ref, total_ref_correct, total_ref_positive = 0, 0, 0, 0
        total_partner_ref_loss, total_partner_ref, total_partner_ref_correct, total_partner_ref_positive = 0, 0, 0, 0

        total_next_mention_loss, total_next_mention, total_next_mention_correct, total_next_mention_positive = 0, 0, 0, 0

        total_attn_ref_stats = {}

        total_word_attn_loss = 0
        total_feed_attn_loss = 0

        for batch in tqdm.tqdm(dataset, ncols=80) if use_tqdm else dataset:
            # for batch in trainset:
            # lang_loss, ref_loss, ref_correct, ref_total, sel_loss, word_attn_loss, feed_attn_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = batch_fn(batch)
            forward_ret = batch_fn(batch)
            total_lang_loss += forward_ret.lang_loss.item()
            total_select_loss += forward_ret.sel_loss.item()
            total_select_correct += forward_ret.sel_correct
            total_select += forward_ret.sel_total

            total_ref_loss += unwrap(forward_ret.ref_loss)
            total_ref_correct += forward_ret.ref_correct
            total_ref += forward_ret.ref_total
            total_ref_positive += forward_ret.ref_positive

            total_partner_ref_loss += unwrap(forward_ret.partner_ref_loss)
            total_partner_ref_correct += forward_ret.partner_ref_correct
            total_partner_ref += forward_ret.partner_ref_total
            total_partner_ref_positive += forward_ret.partner_ref_positive

            total_word_attn_loss += unwrap(forward_ret.word_attn_loss)
            total_feed_attn_loss += unwrap(forward_ret.feed_attn_loss)

            total_next_mention_loss += unwrap(forward_ret.next_mention_loss)
            total_next_mention_correct += forward_ret.next_mention_correct
            total_next_mention += forward_ret.next_mention_total
            total_next_mention_positive += forward_ret.next_mention_positive

            total_attn_ref_stats = utils.sum_dicts(total_attn_ref_stats, forward_ret.attn_ref_stats)

        print("{} total_ref_positive: {}".format(name, total_ref_positive))
        print("{} total_ref_correct/total_ref: {}/{} {:.4f}".format(
            name, total_ref_correct, total_ref, total_ref_correct / total_ref
        ))
        print()

        print("{} total_partner_ref_positive: {}".format(name, total_partner_ref_positive))
        print("{} total_partner_ref_correct/total_partner_ref: {}/{} {:.4f}".format(
            name, total_partner_ref_correct, total_partner_ref, (total_partner_ref_correct / total_partner_ref) if total_partner_ref > 0 else 0
        ))
        print()

        print("{} word_attn_loss: {:.4f}".format(name, total_word_attn_loss))
        print("{} feed_attn_loss: {:.4f}".format(name, total_feed_attn_loss))

        pprint.pprint({'{}_{}'.format(name, k): v for k, v in total_attn_ref_stats.items()})

        total_lang_loss /= len(dataset)
        total_select_loss /= len(dataset)
        total_ref_loss /= len(dataset)
        total_partner_ref_loss /= len(dataset)
        total_next_mention_loss /= len(dataset)
        time_elapsed = time.time() - start_time

        metrics = {
            'lang_loss': total_lang_loss,
            'ref_loss': total_ref_loss,
            'ref_accuracy': total_ref_correct / total_ref,
            'partner_ref_loss': total_partner_ref_loss,
            'partner_ref_accuracy': (total_partner_ref_correct / total_partner_ref) if total_partner_ref > 0 else 0,
            'next_mention_loss': total_next_mention_loss,
            'next_mention_accuracy': (total_next_mention_correct / total_next_mention) if total_next_mention > 0 else 0,
            'select_loss': total_select_loss,
            'select_accuracy': total_select_correct / total_select,
            'time': time_elapsed
        }
        return metrics

    def train_pass(self, trainset, trainset_stats):
        '''
        basic implementation of one training pass
        '''
        self.model.train()
        return self._pass(trainset, self.train_batch, "train", use_tqdm=True)


    def valid_pass(self, validset, validset_stats):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()
        return self._pass(validset, self.valid_batch, "val", use_tqdm=False)


    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_metrics = self.train_pass(trainset, trainset_stats)
        valid_metrics = self.valid_pass(validset, validset_stats)

        if self.verbose:
            print('epoch %03d \t s/epoch %.2f \t lr %.2E' % (epoch, train_metrics['time'], lr))
            print('epoch %03d \t train_lang_loss(scaled) %.4f \t train_ppl %.4f' % (
                epoch, train_metrics['lang_loss'] * self.args.lang_weight, np.exp(train_metrics['lang_loss'])))
            print('epoch %03d \t train_select_loss(scaled) %.4f \t train_select_acc %.4f' % (
                epoch, train_metrics['select_loss'] * self.args.sel_weight, train_metrics['select_accuracy']))
            print('epoch %03d \t train_ref_loss(scaled) %.4f \t train_ref_acc %.4f' % (
                epoch, train_metrics['ref_loss'] * self.args.ref_weight, train_metrics['ref_accuracy']))
            print('epoch %03d \t train_partner_ref_loss(scaled) %.4f \t train_partner_ref_acc %.4f' % (
                epoch, train_metrics['partner_ref_loss'] * self.args.partner_ref_weight, train_metrics['partner_ref_accuracy']))
            print('epoch %03d \t train_next_mention_loss(scaled) %.4f \t train_next_mention_acc %.4f' % (
                epoch, train_metrics['next_mention_loss'] * self.args.next_mention_weight, train_metrics['next_mention_accuracy']))

            print('epoch %03d \t valid_lang_loss %.4f \t valid_ppl %.4f' % (
                epoch, valid_metrics['lang_loss'], np.exp(valid_metrics['lang_loss'])))
            print('epoch %03d \t valid_select_loss %.4f \t valid_select_acc %.4f' % (
                epoch, valid_metrics['select_loss'], valid_metrics['select_accuracy']))
            print('epoch %03d \t valid_ref_loss %.4f \t valid_ref_acc %.4f' % (
                epoch, valid_metrics['ref_loss'], valid_metrics['ref_accuracy']))
            print('epoch %03d \t valid_partner_ref_loss %.4f \t valid_partner_ref_acc %.4f' % (
                epoch, valid_metrics['partner_ref_loss'], valid_metrics['partner_ref_accuracy']))
            print('epoch %03d \t valid_next_mention_loss %.4f \t valid_next_mention_acc %.4f' % (
                epoch, valid_metrics['next_mention_loss'], valid_metrics['next_mention_accuracy']))
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
        return metrics['lang_loss'] * self.args.lang_weight \
               + metrics['select_loss'] * self.args.sel_weight \
               + metrics['ref_loss'] * self.args.ref_weight \
               + metrics['partner_ref_loss'] * self.args.partner_ref_weight


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
                    "update best model: valid_lang_loss %.4f \t valid_select_loss %.4f \t valid_select_acc %.4f \t valid_ref_loss %.4f " %
                    (metrics['valid']['lang_loss'], metrics['valid']['select_loss'], metrics['valid']['select_accuracy'], metrics['valid']['ref_loss'])
                )
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

                # utils.save_model(best_model, model_filename_fn('ep-{}'.format(epoch), 'th'))
                # utils.save_model(best_model.state_dict(), model_filename_fn('ep-{}'.format(epoch), 'stdict'))


        return best_combined_valid_loss, best_model


class HierarchicalRnnReferenceEngine(RnnReferenceEngine):
    @classmethod
    def add_args(cls, parser):
        # don't need to call super because its arguments will already be registered by engines.add_engine_args
        pass

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

    def _ref_loss(self, ref_inpt, ref_tgt, ref_out, num_markables):
        if ref_inpt is None or ref_out is None:
            return None, 0, 0, 0, 0
        ref_tgt = Variable(ref_tgt)
        # max(this_num_markables) x batch_size x num_dots
        ref_mask = torch.zeros_like(ref_tgt)
        for i, nm in enumerate(num_markables):
            ref_mask[i, :nm, :] = 1
        ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
        # print(ref_tgt.size())
        ref_mask = torch.transpose(ref_mask, 0, 1).contiguous()
        assert ref_tgt.size() == ref_out.size()
        assert ref_tgt.size() == ref_mask.size()
        # print('ref_out size: {}'.format(ref_out.size()))
        # print('ref_tgt size: {}'.format(ref_tgt.size()))
        ref_loss = (self.ref_crit_no_reduce(ref_out, ref_tgt) * ref_mask.float()).sum()
        ref_correct = (((ref_out > 0).long() == ref_tgt.long()) * ref_mask.byte()).sum().item()
        ref_total = ref_mask.sum().item()
        ref_gold_positive = ref_tgt.sum().item()
        ref_pred_positive = ((ref_out > 0) * ref_mask.byte()).sum().item()

        return ref_loss, ref_correct, ref_total, ref_gold_positive, ref_pred_positive


    def _forward(self, batch):
        if self.args.word_attention_supervised or self.args.feed_attention_supervised or self.args.mark_dots_mentioned:
            assert self.args.lang_only_self
        ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, scenario_ids, real_ids, partner_real_ids, _, _, sel_idx, lens, rev_idxs, hid_idxs, num_markables, is_self, partner_ref_inpts, partner_ref_tgts_our_view, all_partner_num_markables = batch

        ctx = Variable(ctx)
        bsz = ctx.size(0)
        num_dots = int(ctx.size(1) / 4)
        assert num_dots == 7

        inpts = [Variable(inpt) for inpt in inpts]
        ref_inpts = [Variable(ref_inpt) if ref_inpt is not None else None
                     for ref_inpt in ref_inpts]
        tgts = [Variable(tgt) for tgt in tgts]
        rev_idxs = [Variable(idx) for idx in rev_idxs]
        hid_idxs = [Variable(idx) for idx in hid_idxs]

        # inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables, = self._append_pad(inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables)

        # if single_sent_generation_beliefs is not None:
        #     generation_beliefs = [single_sent_generation_beliefs] * len(inpts)
        # else:
        #     generation_beliefs = None

        last_partner_ref_out = None

        dots_mentioned = make_dots_mentioned_multi(ref_tgts, self.args, bsz, num_dots)
        partner_dots_mentioned_our_view = make_dots_mentioned_multi(partner_ref_tgts_our_view, self.args, bsz, num_dots)

        def belief_function(timestep, partner_ref_outs):
            if timestep >= 0:
                assert len(partner_ref_outs) == timestep
            selection_beliefs, generation_beliefs, mention_beliefs = _make_beliefs(
                bsz, num_dots, self.args, inpts, ref_tgts, partner_ref_tgts_our_view, real_ids, partner_real_ids, sel_tgt, is_self,
                partner_ref_outs,
                timestep,
                partner_dots_mentioned_our_view,
                dots_mentioned,
            )
            return selection_beliefs, generation_beliefs, mention_beliefs

        outs, ref_outs_and_partner_ref_outs, sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_outs = self.model.forward(
            ctx, inpts, ref_inpts, sel_idx, lens, dots_mentioned,
            belief_function=belief_function,
            partner_ref_inpts=partner_ref_inpts,
        )

        sel_tgt = Variable(sel_tgt)
        lang_losses = []
        assert len(inpts) == len(tgts) == len(outs)
        for i, (out, tgt) in enumerate(zip(outs, tgts)):
            # T x bsz
            loss = self.crit_no_reduce(out, tgt).view(-1, bsz)
            if self.args.lang_only_self:
                # loss = loss * (this_is_self.unsqueeze(0).expand_as(loss))
                loss = loss * (is_self[i].unsqueeze(0).expand_as(loss))
            lang_losses.append(loss.sum())
        total_lens = sum(l.sum() for l in lens)
        lang_loss = sum(lang_losses) / total_lens

        ref_correct = 0
        ref_total = 0
        ref_gold_positive = 0
        ref_pred_positive = 0
        ref_losses = []

        partner_ref_correct = 0
        partner_ref_total = 0
        partner_ref_gold_positive = 0
        partner_ref_pred_positive = 0
        partner_ref_losses = []

        attn_ref_true_positive = 0
        attn_ref_total = 0
        attn_ref_gold_positive = 0
        attn_ref_pred_positive = 0

        word_attn_losses = []
        feed_attn_losses = []

        assert len(ref_inpts) == len(ref_tgts) == len(num_markables)

        assert len(partner_ref_inpts) == len(partner_ref_tgts_our_view) == len(all_partner_num_markables)

        # TODO: just index into the lists; the safety check isn't worth it
        for ref_inpt, partner_ref_inpt, (ref_out, partner_ref_out), ref_tgt, partner_ref_tgt, this_num_markables, this_partner_num_markables, this_ctx_attn_prob, this_feed_ctx_attn_prob, this_dots_mentioned, inpt, tgt in utils.safe_zip(
                ref_inpts, partner_ref_inpts, ref_outs_and_partner_ref_outs, ref_tgts, partner_ref_tgts_our_view, num_markables, all_partner_num_markables, ctx_attn_prob, feed_ctx_attn_prob, dots_mentioned, inpts, tgts
        ):
            if (this_num_markables == 0).all() or ref_tgt is None:
                continue
            assert max(this_num_markables) == ref_tgt.size(1)
            _ref_loss, _ref_correct, _ref_total, _ref_gold_positive, _ref_pred_positive = self._ref_loss(
                ref_inpt, ref_tgt, ref_out, this_num_markables
            )
            ref_correct += _ref_correct
            ref_total += _ref_total
            ref_gold_positive += _ref_gold_positive
            ref_pred_positive += _ref_pred_positive
            ref_losses.append(_ref_loss)

            _partner_ref_loss, _partner_ref_correct, _partner_ref_total, _partner_ref_gold_positive, _partner_ref_pred_positive = self._ref_loss(
                partner_ref_inpt, partner_ref_tgt, partner_ref_out, this_partner_num_markables
            )
            partner_ref_correct += _partner_ref_correct
            partner_ref_total += _partner_ref_total
            partner_ref_gold_positive += _partner_ref_gold_positive
            partner_ref_pred_positive += _partner_ref_pred_positive
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
                        assert markable_start > 0 # these are indexes into inpt; we need to subtact one to get indices into tgt
                        pred_dots = set()
                        for t in range(markable_start, markable_end + 1):
                            # indices into the original tcap; i.e. dot indices
                            this_pred_dots = sorted_ix[t, batch_ix][attended[t-1, batch_ix]]
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
                            referent_attention = this_ctx_attn_prob[markable_start-1:markable_end,batch_ix]
                            # kl_div takes inputs as log probabilities, and target probabilities
                            if self.args.attention_supervision_method == 'kl':
                                attn_loss = torch.nn.functional.kl_div(
                                    referent_attention.log(),
                                    gold_dist.unsqueeze(0).expand_as(referent_attention),
                                    reduction='batchmean'
                                )
                            elif self.args.attention_supervision_method == 'penalize_unmentioned':
                                # attn_loss = referent_attention[gold_dist == 0].log().sum()
                                attn_loss = referent_attention[(gold_dist == 0).unsqueeze(0).expand_as(referent_attention)].sum()
                            else:
                                raise ValueError("invalid --attention_supervision_method {}".format(self.args.attention_supervision_method))

                            if attn_loss != attn_loss:
                                print("nan loss: {}\nreferent_attention: {} \t gold_dist: {}".format(attn_loss.item(), referent_attention.log(), gold_dist))
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
                    raise ValueError("invalid --attention_supervision_method {}".format(self.args.attention_supervision_method))
                if feed_attn_loss != feed_attn_loss:
                    # nan
                    print("feed nan loss: {}\nthis_feed_ctx_attn_prob: {} \t this_dots_mentioned: {}".format(
                        feed_attn_loss.item(), this_feed_ctx_attn_prob.log(), this_dots_mentioned
                    ))
                else:
                    feed_attn_losses.append(feed_attn_loss)
        ref_loss = sum(ref_losses) / ref_total

        if partner_ref_total == 0 or (not partner_ref_losses):
            assert partner_ref_total == 0 and (not partner_ref_losses)
            partner_ref_loss = None
        else:
            partner_ref_loss = sum(partner_ref_losses) / partner_ref_total

        # print('sel_out.size(): {}'.format(sel_out.size()))
        # print('sel_tgt.size(): {}'.format(sel_tgt.size()))

        sel_loss = self.sel_crit(sel_out, sel_tgt)
        sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_out.size(0)

        # print("ref_gold_positive: {}".format(ref_gold_positive))
        # print("ref_pred_positive: {}".format(ref_pred_positive))
        # print("ref_correct: {}".format(ref_correct))
        # print("ref_total: {}".format(ref_total))
        #
        # print("attn_ref_gold_positive: {}".format(attn_ref_gold_positive))
        # print("attn_ref_pred_positive: {}".format(attn_ref_pred_positive))
        # print("attn_ref_true_positive: {}".format(attn_ref_true_positive))
        # print("attn_ref_total: {}".format(ref_total))

        attn_ref_stats = {
            'gold_positive': attn_ref_gold_positive,
            'pred_positive': attn_ref_pred_positive,
            'true_positive': attn_ref_true_positive,
            'total': attn_ref_total,
        }

        if word_attn_losses:
            word_attn_loss = sum(word_attn_losses) / len(word_attn_losses)
        else:
            word_attn_loss = None

        if feed_attn_losses:
            feed_attn_loss = sum(feed_attn_losses) / len(feed_attn_losses)
        else:
            feed_attn_loss = None

        next_mention_correct = 0
        next_mention_total = 0
        next_mention_gold_positive = 0
        next_mention_pred_positive = 0
        next_mention_losses = []

        if self.args.next_mention_prediction:
            assert len(dots_mentioned) + 1 == len(next_mention_outs)
            for i in range(len(dots_mentioned)):
                # supervise only for self, so pseudo_num_mentions = 1 iff is_self; 0 otherwise
                pseudo_num_mentions = is_self[i].long()
                gold_dots_mentioned = dots_mentioned[i].long().unsqueeze(1)
                pred_dots_mentioned_logits = next_mention_outs[i]

                # hack; pass True for inpt because this method only uses it to ensure it's not null
                _loss, _correct, _total, _gold_positive, _pred_positive = self._ref_loss(
                    True, gold_dots_mentioned, pred_dots_mentioned_logits, pseudo_num_mentions
                )
                # print("i: {}\tgold_dots_mentioned.sum(): {}\t(pred_dots_mentioned > 0).sum(): {}".format(i, gold_dots_mentioned.sum(), (pred_dots_mentioned > 0).sum()))
                # print("{} / {}".format(_correct, _total))
                next_mention_losses.append(_loss)
                next_mention_correct += _correct
                next_mention_total += _total
                next_mention_gold_positive += _gold_positive
                next_mention_pred_positive += _pred_positive

        if next_mention_total == 0 or (not next_mention_losses):
            assert next_mention_total == 0 and (not next_mention_losses)
            next_mention_loss = None
        else:
            next_mention_loss = sum(next_mention_losses) / next_mention_total

        return ForwardRet(
            lang_loss=lang_loss,
            ref_loss=ref_loss,
            ref_correct=ref_correct,
            sel_loss=sel_loss,
            word_attn_loss=word_attn_loss,
            feed_attn_loss=feed_attn_loss,
            sel_correct=sel_correct,
            sel_total=sel_total,
            ref_positive=ref_gold_positive,
            ref_total=ref_total,
            attn_ref_stats=attn_ref_stats,
            partner_ref_loss=partner_ref_loss,
            partner_ref_correct=partner_ref_correct,
            partner_ref_positive=partner_ref_gold_positive,
            partner_ref_total=partner_ref_total,
            next_mention_loss=next_mention_loss,
            next_mention_correct=next_mention_correct,
            next_mention_positive=next_mention_gold_positive,
            next_mention_total=next_mention_total,
        )
