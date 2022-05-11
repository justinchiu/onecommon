import argparse
import json
import os
import pdb
import re
import random

import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

from agent import *
from belief_agent import BeliefAgent
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger, HierarchicalDialog
from models.rnn_reference_model import RnnReferenceModel
import domain
from static_dialog import StaticHierarchicalDialog, StaticDialogLogger

import pprint

def dump_json(file, path):
    try:
        with open(path, "w") as fout:
            json.dump(file, fout, indent=4, sort_keys=True)
    except:
        raise Exception('Error writing JSON to %s' % path)

class SelfPlay(object):
    def __init__(self, dialog, ctx_gen, args, logger=None, max_n=1000):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()
        self.max_n = max_n

    def run(self):
        max_n = self.max_n
        n = 0
        success = 0
        for ctxs in self.ctx_gen.iter():
            n += 1
            if self.args.smart_alice and n > max_n:
                break
            if n > max_n:
                break
            self.logger.dump('=' * 80)
            self.logger.dump(f'dialog {n}')
            with torch.no_grad():
                _, agree, _ = self.dialog.run(ctxs, self.logger)
            if agree:
                success += 1
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)
        if self.args.plot_metrics:
            self.dialog.plot_metrics()

        return success / n

def get_agent_type(model, smart=False):
    if isinstance(model, (RnnReferenceModel)):
        if smart:
            assert False
        else:
            return RnnAgent
    else:
        assert False, 'unknown model type: %s' % (model)

def make_parser():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--alice_model_file', type=str,
                        help='Alice model file')
    parser.add_argument('--alice_forward_model_file', type=str,
                        help='Alice forward model file')
    parser.add_argument('--bob_model_file', type=str,
                        help='Bob model file')
    parser.add_argument('--context_file', type=str,
                        help='context file')
    # parser.add_argument('--temperature', type=float, default=1.0,
    #                     help='temperature')
    # parser.add_argument('--pred_temperature', type=float, default=1.0,
    #                     help='temperature')
    parser.add_argument('--log_attention', action='store_true', default=False,
                        help='log attention')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print out converations')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--max_turns', type=int, default=20,
                        help='maximum number of turns in a dialog')
    parser.add_argument('--log_file', type=str, default='selfplay.log',
                        help='log dialogs to file')
    parser.add_argument('--markables_file', type=str, default='selfplay_markables.json',
                        help='log markables to file')
    parser.add_argument('--referents_file', type=str, default='selfplay_referents.json',
                        help='log referents to file')
    parser.add_argument('--smart_alice', action='store_true', default=False,
                        help='make Alice smart again')
    parser.add_argument('--rollout_bsz', type=int, default=3,
                        help='rollout batch size')
    parser.add_argument('--rollout_count_threshold', type=int, default=3,
                        help='rollout count threshold')
    parser.add_argument('--smart_bob', action='store_true', default=False,
                        help='make Bob smart again')
    parser.add_argument('--selection_model_file', type=str,  default='',
                        help='path to save the final model')
    parser.add_argument('--rollout_model_file', type=str,  default='',
                        help='path to save the final model')
    parser.add_argument('--ref_text', type=str,
                        help='file with the reference text')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    parser.add_argument('--domain', type=str, default='one_common',
                        help='domain for the dialogue')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='plot graphs')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='eps greedy')
    parser.add_argument('--data', type=str, default='data/onecommon',
                        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=10,
                        help='minimum word frequency to be in dictionary')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch size')
    parser.add_argument('--plot_metrics', action='store_true', default=False,
                        help='plot metrics')
    parser.add_argument('--markable_detector_file', type=str, default="markable_detector",
                        help='visualize referents')
    parser.add_argument('--record_markables', action='store_true', default=False,
                        help='record markables and referents')
    parser.add_argument('--repeat_selfplay', action='store_true', default=False,
                        help='repeat selfplay')
    parser.add_argument('--num_contexts', type=int, default=1000,
                        help='num_contexts')
    parser.add_argument('--must_contain', nargs="*", 
                        help='must contain scenarios')


    RnnAgent.add_args(parser)

    return parser


def main():
    parser = make_parser()

    utils.dump_git_status(sys.stdout)
    print(' '.join(sys.argv))
    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.repeat_selfplay:
        seeds = list(range(10))
    else:
        seeds = [args.seed]

    repeat_results = []

    for seed in seeds:
        utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        def model_filename_fn(model_file, name, extension):
            return '{}_{}_{}.{}'.format(model_file, seed, name, extension)

        assert os.path.exists(args.markable_detector_file)
        markable_detector = utils.load_model(args.markable_detector_file, prefix_dir=None, map_location='cpu')
        if args.cuda:
            markable_detector.cuda()
        markable_detector.eval()
        # markable_detector_corpus = markable_detector.corpus_ty(
        #     domain, args.data, train='train_markable_{}.txt'.format(seed),
        #     valid='valid_markable_{}.txt'.format(seed), test='test_markable_{}.txt'.format(seed), #test='selfplay_reference_{}.txt'.format(seed),
        #     freq_cutoff=args.unk_threshold, verbose=True
        # )


        # alice_model = utils.load_model(args.alice_model_file + '_' + str(seed) + '.th')
        alice_model = utils.load_model(args.alice_model_file, prefix_dir=None, map_location='cpu')
        #alice_ty = get_agent_type(alice_model, args.smart_alice)
        alice_ty = BeliefAgent
        alice_merged_args = argparse.Namespace(**utils.merge_dicts(vars(args), vars(alice_model.args)))
        alice = alice_ty(alice_model, alice_merged_args, name='Alice', train=False, markable_detector=markable_detector)

        # bob_model = utils.load_model(args.bob_model_file + '_' + str(seed) + '.th')
        bob_model = utils.load_model(args.bob_model_file, prefix_dir=None, map_location='cpu')
        #bob_ty = get_agent_type(bob_model, args.smart_bob)
        bob_ty = BeliefAgent
        bob_merged_args = argparse.Namespace(**utils.merge_dicts(vars(args), vars(bob_model.args)))
        bob = bob_ty(bob_model, bob_merged_args, name='Bob', train=False, markable_detector=markable_detector)

        for model in [alice_model, bob_model]:
            if args.cuda:
                model.cuda()
            model.eval()

        train_data_path = "data/onecommon/final_transcripts.json"
        with open(train_data_path, "r") as f:
            dialogues = json.load(f)
        if args.must_contain is not None:
            dialogues = [d for d in dialogues if d["scenario"]["uuid"] in args.must_contain]
        # hack must_contain to include dialogues in final_transcripts
        must_contain = [d["scenario"]["uuid"] for d in dialogues]

        # dialog = Dialog([alice, bob], args, markable_detector)
        dialog = StaticHierarchicalDialog([alice, bob], args, markable_detector, dialogues)
        ctx_gen = ContextGenerator(
            'data/onecommon/static/train_context_9.txt',
            #must_contain = args.must_contain,
            must_contain = must_contain,
        )
        with open(os.path.join('data/onecommon/static/scenarios.json'), "r") as f:
            scenario_list = json.load(f)
        scenarios = {
            scenario['uuid']: scenario
            for scenario in scenario_list
            #if args.must_contain is None or scenario["uuid"] in args.must_contain
            if args.must_contain is None or scenario["uuid"] in must_contain
        }
        logger = DialogLogger(verbose=args.verbose, log_file=args.log_file, scenarios=scenarios)

        selfplay = SelfPlay(dialog, ctx_gen, args, logger, max_n = args.num_contexts)
        result = selfplay.run()
        repeat_results.append(result)


    if args.markables_file:
        print(f"dump {args.markables_file}")
        dump_json(dialog.selfplay_markables, args.markables_file)
    if args.referents_file:
        print(f"dump {args.referents_file}")
        dump_json(dialog.selfplay_referents, args.referents_file)

    print("repeat selfplay results %.8f ( %.8f )" % (np.mean(repeat_results), np.std(repeat_results)))


if __name__ == '__main__':
    main()
