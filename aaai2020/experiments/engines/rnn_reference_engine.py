import copy
import pprint
import time

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

import utils
from engines import EngineBase


class RnnReferenceEngine(EngineBase):
    def __init__(self, model, args, verbose=False):
        super(RnnReferenceEngine, self).__init__(model, args, verbose)

    def _forward(self, batch):
        ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, _, _, _, sel_idx = batch

        ctx = Variable(ctx)
        inpt = Variable(inpt)
        if ref_inpt is not None:
            ref_inpt = Variable(ref_inpt)

        assert ref_tgt.dim() == 3
        dots_mentioned = ref_tgt.sum(1) > 0
        out, ref_out, sel_out, ctx_attn_prob = self.model.forward(ctx, inpt, ref_inpt, sel_idx, lens=None,
                                                                  dots_mentioned=dots_mentioned)

        tgt = Variable(tgt)
        sel_tgt = Variable(sel_tgt)
        lang_loss = self.crit(out, tgt)

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

        sel_loss = self.sel_crit(sel_out, sel_tgt)
        sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_out.size(0)

        # TODO
        attn_ref_stats = {}

        return lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_positive, attn_ref_stats

    def train_batch(self, batch):
        lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = self._forward(
            batch)

        # default
        loss = None
        return_ref_loss = 0
        if self.args.lang_weight > 0:
            loss = self.args.lang_weight * lang_loss
            if self.args.sel_weight > 0:
                loss += self.args.sel_weight * sel_loss
            if self.args.ref_weight > 0 and ref_loss is not None:
                loss += self.args.ref_weight * ref_loss
                return_ref_loss = ref_loss.item()
        elif self.args.sel_weight > 0:
            loss = self.args.sel_weight * sel_loss / sel_total
            if self.args.ref_weight > 0 and ref_loss is not None:
                loss += self.args.ref_weight * ref_loss
                return_ref_loss = ref_loss.item()
        elif self.args.ref_weight > 0 and ref_loss is not None:
            loss = self.args.ref_weight * ref_loss
            return_ref_loss = ref_loss.item()

        if loss:
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()

        return lang_loss.item(), return_ref_loss, ref_correct, ref_total, sel_loss.item(), sel_correct, sel_total, ref_positive, attn_ref_stats

    def valid_batch(self, batch):
        with torch.no_grad():
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = self._forward(
                batch)

        if ref_loss is not None:
            return_ref_loss = ref_loss.item()
        else:
            return_ref_loss = 0

        return lang_loss.item(), return_ref_loss, ref_correct, ref_total, sel_loss.item(), sel_correct, sel_total, ref_positive, attn_ref_stats

    def test_batch(self, batch):
        with torch.no_grad():
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = self._forward(
                batch)

        if ref_loss is not None:
            return_ref_loss = ref_loss.item()
        else:
            return_ref_loss = 0

        return lang_loss.item(), return_ref_loss, ref_correct, ref_total, sel_loss.item(), sel_correct, sel_total, ref_positive, attn_ref_stats

    def train_pass(self, trainset, trainset_stats):
        '''
        basic implementation of one training pass
        '''
        self.model.train()

        total_lang_loss, total_select_loss, total_select, total_select_correct, total_reference_loss, total_reference, total_reference_correct = 0, 0, 0, 0, 0, 0, 0
        total_ref_positive = 0
        start_time = time.time()

        total_attn_ref_stats = {}

        for batch in tqdm.tqdm(trainset, ncols=80):
            # for batch in trainset:
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = self.train_batch(
                batch)
            total_lang_loss += lang_loss
            total_select_loss += sel_loss
            total_select_correct += sel_correct
            total_select += sel_total
            total_reference_loss += ref_loss
            total_reference_correct += ref_correct
            total_reference += ref_total
            total_ref_positive += ref_positive

            total_attn_ref_stats = utils.sum_dicts(total_attn_ref_stats, attn_ref_stats)

        print("train total_ref_positive: {}".format(total_ref_positive))
        print("train total_ref_correct/total_ref: {}/{} {:.4f}".format(
            total_reference_correct, total_reference, total_reference_correct / total_reference
        ))

        pprint.pprint({'train_{}'.format(k): v for k, v in total_attn_ref_stats.items()})

        total_lang_loss /= len(trainset)
        total_select_loss /= len(trainset)
        total_reference_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_lang_loss, total_reference_loss, total_reference_correct / total_reference, total_select_loss, total_select_correct / total_select, time_elapsed

    def valid_pass(self, validset, validset_stats):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()

        total_lang_loss, total_select_loss, total_select, total_select_correct, total_reference_loss, total_reference, total_reference_correct = 0, 0, 0, 0, 0, 0, 0
        total_ref_positive = 0

        total_attn_ref_stats = {}

        for batch in validset:
            lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_positive, attn_ref_stats = self.valid_batch(
                batch)
            total_lang_loss += lang_loss
            total_select_loss += sel_loss
            total_select_correct += sel_correct
            total_select += sel_total
            total_reference_loss += ref_loss
            total_reference_correct += ref_correct
            total_reference += ref_total
            total_ref_positive += ref_positive

            total_attn_ref_stats = utils.sum_dicts(total_attn_ref_stats, attn_ref_stats)

        print("val total_ref_positive: {}".format(total_ref_positive))
        print("val total_ref_correct/total_ref: {}/{} {:.4f}".format(
            total_reference_correct, total_reference, total_reference_correct / total_reference
        ))

        pprint.pprint({'val_{}'.format(k): v for k, v in total_attn_ref_stats.items()})

        total_lang_loss /= len(validset)
        total_select_loss /= len(validset)
        total_reference_loss /= len(validset)
        return total_lang_loss, total_reference_loss, total_reference_correct / total_reference, total_select_loss, total_select_correct / total_select

    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_lang_loss, train_reference_loss, train_reference_accuracy, train_select_loss, train_select_accuracy, train_time = self.train_pass(
            trainset, trainset_stats)
        valid_lang_loss, valid_reference_loss, valid_reference_accuracy, valid_select_loss, valid_select_accuracy = self.valid_pass(
            validset, validset_stats)

        if self.verbose:
            print('epoch %03d \t s/epoch %.2f \t lr %.2E' % (epoch, train_time, lr))
            print('epoch %03d \t train_lang_loss(scaled) %.4f \t train_ppl %.4f' % (
                epoch, train_lang_loss * self.args.lang_weight, np.exp(train_lang_loss)))
            print('epoch %03d \t train_select_loss(scaled) %.4f \t train_select_acc %.4f' % (
                epoch, train_select_loss * self.args.sel_weight, train_select_accuracy))
            print('epoch %03d \t train_ref_loss(scaled) %.4f \t train_ref_acc %.4f' % (
                epoch, train_reference_loss * self.args.ref_weight, train_reference_accuracy))
            print('epoch %03d \t valid_lang_loss %.4f \t valid_ppl %.4f' % (
                epoch, valid_lang_loss, np.exp(valid_lang_loss)))
            print('epoch %03d \t valid_select_loss %.4f \t valid_select_acc %.4f' % (
                epoch, valid_select_loss, valid_select_accuracy))
            print('epoch %03d \t valid_ref_loss %.4f \t valid_ref_acc %.4f' % (
                epoch, valid_reference_loss, valid_reference_accuracy))
            print()

        if self.args.tensorboard_log:
            info = {'Train_Lang_Loss': train_lang_loss,
                    'Train_Select_Loss': train_select_loss,
                    'Valid_Lang_Loss': valid_lang_loss,
                    'Valid_Select_Loss': valid_select_loss,
                    'Valid_Select_Accuracy': valid_select_accuracy}
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, epoch)

            for tag, value in self.model.named_parameters():
                if value.grad is None:
                    continue
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                self.logger.histo_summary(
                    tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        return valid_lang_loss, valid_select_loss, valid_reference_loss, valid_select_accuracy

    def combine_loss(self, lang_loss, select_loss, reference_loss):
        return lang_loss * int(self.args.lang_weight > 0) + select_loss * int(
            self.args.sel_weight > 0) + reference_loss * int(self.args.ref_weight > 0)

    def train(self, corpus, model_filename_fn):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        validdata = corpus.valid_dataset(self.args.bsz)

        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            valid_lang_loss, valid_select_loss, valid_reference_loss, valid_select_acc = self.iter(epoch,
                                                                                                   self.opt.param_groups[
                                                                                                       0]["lr"],
                                                                                                   traindata, validdata)

            if self.scheduler is not None:
                self.scheduler.step(valid_select_loss)

            combined_valid_loss = self.combine_loss(valid_lang_loss, valid_select_loss, valid_reference_loss)
            if combined_valid_loss < best_combined_valid_loss:
                print(
                    "update best model: valid_lang_loss %.4f \t valid_select_loss %.4f \t valid_select_acc %.4f \t valid_ref_loss %.4f " %
                    (valid_lang_loss, valid_select_loss, valid_select_acc, valid_reference_loss))
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

                # utils.save_model(best_model, model_filename_fn('ep-{}'.format(epoch), 'th'))
                # utils.save_model(best_model.state_dict(), model_filename_fn('ep-{}'.format(epoch), 'stdict'))

        return best_combined_valid_loss, best_model


class HierarchicalRnnReferenceEngine(RnnReferenceEngine):
    def _append_pad(self, inpts, ref_inpts, tgts, ref_tgts, lens, rev_idxs, hid_idxs, num_markables):
        # TODO: figure out why FAIR's e2e code had this; call it if necessary
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

    def _forward(self, batch):
        ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, scenario_ids, _, _, _, sel_idx, lens, rev_idxs, hid_idxs, num_markables = batch

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

        dots_mentioned = []
        for ref_tgt in ref_tgts:
            if ref_tgt is None:
                dots_mentioned.append(torch.zeros(bsz, num_dots).bool())
                continue
            assert ref_tgt.dim() == 3
            dots_mentioned.append(ref_tgt.sum(1) > 0)

        outs, ref_outs, sel_out, ctx_attn_prob = self.model.forward(ctx, inpts, ref_inpts, sel_idx, lens,
                                                                    dots_mentioned)

        sel_tgt = Variable(sel_tgt)
        lang_losses = []
        assert len(inpts) == len(tgts) == len(outs)
        for i, (out, tgt) in enumerate(zip(outs, tgts)):
            # print('{} out.size(): {}'.format(i, out.size()))
            # print('{} tgt.size(): {}'.format(i, tgt.size()))
            YOU = self.model.word_dict.word2idx['YOU:']
            THEM = self.model.word_dict.word2idx['THEM:']
            is_self = inpts[i][0] == YOU
            is_other = inpts[i][0] == THEM
            assert is_self.sum() + is_other.sum() == bsz
            crit = self.crit_no_reduce(out, tgt).view(-1, bsz)
            if self.args.lang_only_self:
                crit = crit * (is_self.unsqueeze(0).expand_as(crit))
            lang_losses.append(crit.sum())
        total_lens = sum(l.sum() for l in lens)
        lang_loss = sum(lang_losses) / total_lens

        ref_correct = 0
        ref_total = 0
        ref_gold_positive = 0
        ref_pred_positive = 0
        ref_losses = []

        attn_ref_true_positive = 0
        attn_ref_total = 0
        attn_ref_gold_positive = 0
        attn_ref_pred_positive = 0

        assert len(ref_inpts) == len(ref_tgts) == len(num_markables)
        for ref_inpt, ref_out, ref_tgt, this_num_markables, this_ctx_attn_prob in zip(ref_inpts, ref_outs, ref_tgts,
                                                                                      num_markables, ctx_attn_prob):
            if (this_num_markables == 0).all() or ref_tgt is None:
                continue
            assert max(this_num_markables) == ref_tgt.size(1)
            ref_tgt = Variable(ref_tgt)
            # max(this_num_markables) x batch_size x num_dots
            ref_mask = torch.zeros_like(ref_tgt)
            for i, nm in enumerate(this_num_markables):
                ref_mask[i, :nm, :] = 1
            ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
            # print(ref_tgt.size())
            ref_mask = torch.transpose(ref_mask, 0, 1).contiguous()
            assert ref_tgt.size() == ref_out.size()
            assert ref_tgt.size() == ref_mask.size()
            # print('ref_out size: {}'.format(ref_out.size()))
            # print('ref_tgt size: {}'.format(ref_tgt.size()))
            ref_loss = (self.ref_crit_no_reduce(ref_out, ref_tgt) * ref_mask.float()).sum()
            ref_correct += (((ref_out > 0).long() == ref_tgt.long()) * ref_mask.byte()).sum().item()
            ref_total += ref_mask.sum().item()
            ref_gold_positive += ref_tgt.sum().item()
            ref_pred_positive += ((ref_out > 0) * ref_mask.byte()).sum().item()
            ref_losses.append(ref_loss)

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
                        pred_dots = set()
                        for t in range(markable_start, markable_end + 1):
                            # indices into the original tcap; i.e. dot indices
                            this_pred_dots = sorted_ix[t, batch_ix][attended[t, batch_ix]]
                            pred_dots.update(set(this_pred_dots.cpu().detach().numpy()))
                        gold_pos = set(ref_tgt[markable_ix, batch_ix].nonzero().flatten().cpu().detach().numpy())
                        attn_ref_true_positive += len(pred_dots & gold_pos)
                        attn_ref_gold_positive += len(gold_pos)
                        attn_ref_total += num_dots
                        attn_ref_pred_positive += len(pred_dots)

        ref_loss = sum(ref_losses) / ref_total

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

        return lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, ref_gold_positive, attn_ref_stats
