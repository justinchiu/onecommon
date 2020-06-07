import copy
import os
import random

import numpy as np
import torch

from corpora.data import Dictionary, read_lines, get_tag


class MarkableCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=2, train='train_markable.txt',
                 valid='valid_markable.txt', test='test_markable.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.bio_dict = {"B":0, "I":1, "O":2, "<START>":3, "<STOP>":4, "<PAD>": 5}

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            markable_idxs = [self.bio_dict[val] for val in get_tag(tokens, 'markables')]
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]
            dataset.append((input_vals, word_idxs, markable_idxs, scenario_id, agent, chat_id))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by dialog length and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, markables, scenario_ids, agents, chat_ids = [], [], [], [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                markables.append(dataset[j][2])
                scenario_ids.append(dataset[j][3])
                agents.append(dataset[j][4])
                chat_ids.append(dataset[j][5])
                assert len(words) == len(markables)

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                # one additional pad
                words[j] += [pad] * (max_len - len(words[j]) + 1)
                markables[j] += [self.bio_dict["<PAD>"]] * (max_len - len(markables[j]) + 1)

            # construct tensor for context
            ctx = torch.Tensor(inputs).float().squeeze()
            words = torch.Tensor(words).long().transpose(0, 1).contiguous().squeeze()
            markables = torch.Tensor(markables).long().transpose(0, 1).contiguous().squeeze()

            if device is not None:
                ctx = ctx.to(device)
                words = words.to(device)
                markables = markables.to(device)

            batches.append((ctx, words, markables, scenario_ids, agents, chat_ids))

        if shuffle:
            random.shuffle(batches)

        return batches, stats