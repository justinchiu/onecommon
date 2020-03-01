import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.modules.loss import _Loss

from logger import Logger, TFLogger


def add_training_args(parser):
    group = parser.add_argument_group('training')
    group.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam',
                       help='optimizer to use')
    group.add_argument('--lr', type=float, default=0.001,
                       help='initial learning rate')
    group.add_argument('--min_lr', type=float, default=1e-5,
                       help='min threshold for learning rate annealing')
    group.add_argument('--decay_rate', type=float, default=9.0,
                       help='decrease learning rate by this factor')
    group.add_argument('--decay_every', type=int, default=1,
                       help='decrease learning rate after decay_every epochs')
    group.add_argument('--momentum', type=float, default=0.0,
                       help='momentum for sgd')
    group.add_argument('--clip', type=float, default=0.5,
                       help='gradient clipping')
    group.add_argument('--dropout', type=float, default=0.5,
                       help='dropout rate in embedding layer')
    group.add_argument('--init_range', type=float, default=0.01,
                       help='initialization range')
    group.add_argument('--max_epoch', type=int, default=20,
                       help='max number of epochs')
    group.add_argument('--bsz', type=int, default=16,
                       help='batch size')

    group.add_argument('--reduce_plateau', action='store_true')
    group.add_argument('--reduce_plateau_factor', type=float, default=0.2)
    group.add_argument('--reduce_plateau_patience', type=float, default=4)
    group.add_argument('--reduce_plateau_min_lr', type=float, default=1e-6)

def add_engine_args(parser):
    from engines.rnn_reference_engine import RnnReferenceEngine, HierarchicalRnnReferenceEngine
    for eng in [RnnReferenceEngine, HierarchicalRnnReferenceEngine]:
        eng.add_args(parser)

class Criterion(object):
    """Weighted CrossEntropyLoss."""

    def __init__(self, dictionary, device_id=None, bad_toks=[], reduction='mean'):
        w = torch.Tensor(len(dictionary)).fill_(1)
        for tok in bad_toks:
            w[dictionary.get_idx(tok)] = 0.0
        if device_id is not None:
            w = w.cuda(device_id)
        # https://pytorch.org/docs/stable/nn.html
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction)

    def __call__(self, out, tgt):
        return self.crit(out, tgt)


class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * torch.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * torch.sum(loss, dim=1)
        avg_kl_loss = torch.mean(kl_loss)
        return avg_kl_loss


class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        qy = torch.exp(log_qy)
        y_kl = torch.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return torch.mean(y_kl)
        else:
            return torch.sum(y_kl) / batch_size


class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return torch.mean(h_q)
        else:
            return torch.sum(h_q) / batch_size


class EngineBase(object):
    """Base class for training engine."""

    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, model, args, verbose=False):
        self.model = model
        self.args = args
        self.verbose = verbose
        self.opt, self.scheduler = self.make_opt()
        self.crit = Criterion(self.model.word_dict, bad_toks=['<pad>'])
        self.crit_no_reduce = Criterion(self.model.word_dict, bad_toks=['<pad>'], reduction='none')
        self.sel_crit = nn.CrossEntropyLoss(reduction='mean')
        self.ref_crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.ref_crit_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
        if args.tensorboard_log:
            log_name = 'tensorboard_logs/{}'.format(args.model_type)
            if os.path.exists(log_name):
                print("remove old tensorboard log")
                shutil.rmtree(log_name)
            self.logger = TFLogger(log_name)
        else:
            self.logger = Logger()

    def make_opt(self):
        if self.args.optimizer == 'adam':
            opt = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr)
        elif self.args.optimizer == 'rmsprop':
            opt = optim.RMSprop(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum)
        else:
            raise ValueError('invalid optimizer {}'.format(self.args.optimzier))
        if self.args.reduce_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=self.args.reduce_plateau_factor,
                verbose=True,
                patience=self.args.reduce_plateau_patience,
                min_lr=self.args.min_lr,
                threshold=1e-5,
            )
        else:
            scheduler = None

        return opt, scheduler

    def get_model(self):
        return self.model

    def train_batch(self, batch):
        pass

    def valid_batch(self, batch):
        pass

    def train_pass(self, trainset, trainset_stats):
        '''
        basic implementation of one training pass
        '''
        self.model.train()

        total_lang_loss, total_select_loss, total_num_correct, total_num_select = 0, 0, 0, 0
        start_time = time.time()

        for batch in trainset:
            lang_loss, select_loss, num_correct, num_select = self.train_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += select_loss
            total_num_correct += num_correct
            total_num_select += num_select

        total_lang_loss /= len(trainset)
        total_select_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_lang_loss, total_select_loss, total_num_correct / total_num_select, time_elapsed

    def valid_pass(self, validset, validset_stats):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()

        total_lang_loss, total_select_loss, total_num_correct, total_num_select = 0, 0, 0, 0
        for batch in validset:
            lang_loss, select_loss, num_correct, num_select = self.valid_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += select_loss
            total_num_correct += num_correct
            total_num_select += num_select

        total_lang_loss /= len(validset)
        total_select_loss /= len(validset)
        return total_lang_loss, total_select_loss, total_num_correct / total_num_select

    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_lang_loss, train_select_loss, train_select_accuracy, train_time = self.train_pass(trainset,
                                                                                                trainset_stats)
        valid_lang_loss, valid_select_loss, valid_select_accuracy = self.valid_pass(validset, validset_stats)

        if self.verbose:
            print('epoch %03d \t s/epoch %.2f \t lr %.2E' % (epoch, train_time, lr))
            print('epoch %03d \t train_lang_loss(scaled) %.4f \t train_ppl %.4f' % (
                epoch, train_lang_loss * self.args.lang_weight, np.exp(train_lang_loss)))
            print('epoch %03d \t train_select_loss(scaled) %.4f \t train_select_acc %.4f' % (
                epoch, train_select_loss * self.args.sel_weight, train_select_accuracy))
            print('epoch %03d \t valid_lang_loss %.4f \t valid_ppl %.4f' % (
                epoch, valid_lang_loss, np.exp(valid_lang_loss)))
            print('epoch %03d \t valid_select_loss %.4f \t valid_select_acc %.4f' % (
                epoch, valid_select_loss, valid_select_accuracy))
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

        return valid_lang_loss, valid_select_loss

    def combine_loss(self, lang_loss, select_loss):
        return lang_loss + select_loss * self.args.sel_weight

    def train(self, corpus, model_filename_fn):
        raise NotImplementedError()
