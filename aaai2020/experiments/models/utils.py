# Copyright 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A set of useful tools.
"""

import torch
import torch.nn as nn
import math


def init_rnn(rnn, init_range, weights=None, biases=None):
    """ Orthogonal initialization of RNN. """
    return
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    bound = 1 / math.sqrt(rnn.hidden_size)

    # init weights
    for w in weights:
        #nn.init.orthogonal(rnn._parameters[w])
        rnn._parameters[w].data.uniform_(-bound, bound)
        #rnn._parameters[w].data.uniform_(-init_range, init_range)
        #rnn._parameters[w].data.orthogonal_()
    # init biases
    for b in biases:
        p = rnn._parameters[b]
        n = p.size(0)
        p.data.fill_(0)
        # init bias for the reset gate in GRU
        p.data.narrow(0, 0, n // 3).fill_(0.0)


def init_rnn_cell(rnn, init_range):
    """ Orthogonal initialization of RNNCell. """
    return
    init_rnn(rnn, init_range, ['weight_ih', 'weight_hh'], ['bias_ih', 'bias_hh'])


def init_linear(linear, init_range):
    """ Uniform initialization of Linear. """
    return
    linear.weight.data.uniform_(-init_range, init_range)
    linear.bias.data.fill_(0)


def init_cont(cont, init_range):
    """ Uniform initialization of a container. """
    return
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)


def make_mask(n, marked, value=-1000):
    """ Create a masked tensor. """
    mask = torch.Tensor(n).fill_(0)
    for i in marked:
        mask[i] = value
    return mask


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

def unsigmoid(x):
    # aka logit or link; unsigmoid(x).sigmoid() == x
    return (x / (1.0 - x)).log()

def lengths_to_mask(max_length, lengths):
    # 1 for positions up to the length; 0 for others
    return torch.arange(max_length, device=lengths.device).unsqueeze(0).expand(lengths.size(0), -1) < lengths.unsqueeze(1)