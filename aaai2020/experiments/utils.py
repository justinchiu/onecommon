# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Various helpers.
"""

import os
import random
import copy
import pdb
import sys
import json

import torch
import numpy as np
import subprocess

from contextlib import contextmanager


def backward_hook(grad):
    """Hook for backward pass."""
    print(grad)
    pdb.set_trace()
    return grad

MODEL_DIR = 'serialized_models'

def save_model(model, file_name):
    """Serializes model to a file."""
    if file_name != '':
        with open(os.path.join(MODEL_DIR, file_name), 'wb') as f:
            torch.save(model, f)


def load_model(file_name, map_location=None):
    """Reads model from a file."""
    with open(os.path.join(MODEL_DIR, file_name), 'rb') as f:
        if map_location is not None:
            return torch.load(f, map_location=map_location)
        else:
            return torch.load(f)

def sum_dicts(d1, d2):
    ret = {}
    for key in set(d1.keys()) | set(d2.keys()):
        v = d1.get(key, 0.0) + d2.get(key, 0.0)
        ret[key] = v
    return ret

def safe_zip(*lists):
    for l in lists[1:]:
        assert len(l) == len(lists[0])
    return zip(*lists)

def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def use_cuda(enabled, device_id=0):
    """Verifies if CUDA is available and sets default device to be device_id."""
    if not enabled:
        return None
    if not torch.cuda.is_available():
        print('CUDA is not available')
        return None
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_id)
    return device_id

@contextmanager
def set_temporary_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor

    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)

def prob_random():
    """Prints out the states of various RNGs."""
    print('random state: python %.3f torch %.3f numpy %.3f' % (
        random.random(), torch.rand(1)[0], np.random.rand()))


class ContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file."""

    def __init__(self, context_file):
        self.ctxs = []
        with open(context_file, 'r') as f:
            ctx_data = []
            for line in f:
                ctx = line.strip().split()
                ctx_data.append(ctx)
                if len(ctx_data) == 5:
                    self.ctxs.append(ctx_data)
                    ctx_data = []

    def sample(self):
        ctx_data = random.choice(self.ctxs)
        return ctx_data[0], ctx_data[1:3], ctx_data[3:]

    def iter(self, nepoch=1):
        for e in range(nepoch):
            random.shuffle(self.ctxs)
            for ctx_data in self.ctxs:
                yield ctx_data[0], ctx_data[1:3], ctx_data[3:]


class ContextTestGenerator(object):
    def __init__(self, context_file, test_type):
        if test_type == 'unary':
            self.ctxs = []
            with open(context_file, 'r') as f:
                ctx_data = []
                for line in f:
                    ctx = line.strip().split()
                    ctx_data.append(ctx)
                    if len(ctx_data) == 5:
                        self.ctxs.append(ctx_data)
                        ctx_data = []

        self.ctxs = []
        with open(context_file, 'r') as f:
            ctx_data = []
            for line in f:
                ctx = line.strip().split()
                ctx_data.append(ctx)
                if len(ctx_data) == 5:
                    self.ctxs.append(ctx_data)
                    ctx_data = []

def dump_git_status(out_file=sys.stdout, exclude_file_patterns=['*.ipynb', '*.th', '*.sh', '*.txt', '*.json']):
    subprocess.call('git rev-parse HEAD', shell=True, stdout=out_file)
    exclude_string = ' '.join("':(exclude){}'".format(f) for f in exclude_file_patterns)
    subprocess.call('git --no-pager diff -- . {}'.format(exclude_string), shell=True, stdout=out_file)
