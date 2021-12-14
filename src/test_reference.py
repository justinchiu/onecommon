"""
Performs evaluation of the model on the test dataset.
"""

import argparse
import copy
import json
import os
import re
from collections import Counter, defaultdict
import pprint
import sys

import tqdm

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from corpora import data
from corpora.reference import ReferenceCorpus
from corpora.reference_sentence import ReferenceSentenceCorpus
import utils
from engines import Criterion
from domain import get_domain

import matplotlib.pyplot as plt
import seaborn as sns

from models.reference_predictor import ReferencePredictor, PragmaticReferencePredictor, RerankingMentionPredictor
from engines.rnn_reference_engine import make_dots_mentioned_multi, make_dots_mentioned_per_ref_multi, \
    CAN_CONFIRM_VALUES, make_can_confirm
from engines.rnn_reference_engine import add_metrics, flatten_metrics
from engines.rnn_reference_engine import add_selection_stats
from engines.beliefs import BeliefConstructor, BlankBeliefConstructor

sns.set(font_scale=1.15)

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def dump_json(file, path):
    try:
        with open(path, "w") as fout:
            json.dump(file, fout, indent=4, sort_keys=True)
    except:
        raise Exception('Error writing JSON to %s' % path)

def reference_to_svg(kb, ref_out):
    svg = '''<svg id="svg" width="430" height="430"><circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>\n'''
    for i, obj in enumerate(kb):
        svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"{3}\" />\n".format(obj['x'], obj['y'],
                                                                                             obj['size'], obj['color'])
        if ref_out[i] == 1:
            svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"none\" stroke=\"{3}\" stroke-width=\"4\" stroke-dasharray=\"3,3\"\n/>".format(obj['x'], obj['y'],
                        obj['size'] + 4, "green")
    svg += '''</svg>'''
    return svg

DEFAULT_STATS = [
    'select_acc',
    'ref_accuracy', 'ref_f1', 'ref_exact_match',
    'partner_ref_accuracy', 'partner_ref_f1', 'partner_ref_exact_match',
    'next_mention_accuracy', 'next_mention_f1', 'next_mention_exact_match'
]

def main():
    parser = argparse.ArgumentParser(description='testing script for reference resolution')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--max_instances_per_split', type=int)

    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--model_dir', type=str,
        help='directory containing {split}_best.th')
    parser.add_argument('--model_file', type=str,
                        help='full model file (should end with {split}_best.th or {split}_ep-*.th) \
                        [for backward compatibility with non-directory-based saving]')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--vocab_corpus', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='vocabulary of the corpus to use')
    parser.add_argument('--corpus_type', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='type of test corpus to use')
    parser.add_argument('--bleu_n', type=int, default=0,
        help='test ngram bleu')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature')
    parser.add_argument('--repeat_test', action='store_true', default=False,
        help='repeat training n times')

    parser.add_argument('--eval_split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--eval_split_num', type=int, default=1)

    # for error analysis
    parser.add_argument('--transcript_file', type=str, default='final_transcripts.json',
        help='scenario file')
    parser.add_argument('--markable_file', type=str, default='markable_annotation.json',
        help='scenario file')
    parser.add_argument('--show_errors', action='store_true', default=False,
        help='show errors')

    parser.add_argument('--model_referent_annotation_output_path', help='write the referent predictions to this file')

    parser.add_argument(
        '--lang_only_self',
        action='store_true',
        help='if passed, compute perplexity over both players\' utterances'
    )

    parser.add_argument(
        '--allow_belief_cheating',
        action='store_true',
        help='pass dots_mentioned and selection_beliefs and generation_beliefs'
    )

    parser.add_argument(
        '--reference_prediction', choices=['l0', 'l1'], default='l0'
    )
    parser.add_argument(
        '--partner_reference_prediction', choices=['l0', 'l1'], default='l0'
    )
    PragmaticReferencePredictor.add_args(parser)

    parser.add_argument('--next_mention_reranking', action='store_true')
    parser.add_argument('--next_mention_reranking_score',
                        choices=['entropy_reduction', 'log_max_probability'], default='log_max_probability')
    parser.add_argument('--next_mention_reranking_k', type=int, default=10)
    parser.add_argument('--next_mention_reranking_weights', type=float, nargs='*',
                        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--next_mention_reranking_default_weight', type=float, default=0.0)
    parser.add_argument('--force_next_mention_num_markables', action='store_true')

    parser.add_argument('--selection_confidence', action='store_true')
    parser.add_argument('--selection_confidence_look_ahead', action='store_true')
    parser.add_argument('--selection_confidence_weight', type=float, default=1.0)
    parser.add_argument('--selection_confidence_threshold', type=float, default=0.5)

    utils.dump_git_status(sys.stdout)
    print(' '.join(sys.argv))
    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.bleu_n > 0:
        # current support
        args.bsz = 1

    assert args.model_dir or args.model_file
    assert not (args.model_dir and args.model_file), f"can't pass both args.model_dir and args.model_file; {args.model_dir}, {args.model_file}"

    if args.repeat_test:
        assert args.model_dir
        splits = list(range(10))
    else:
        assert args.model_file
        # assert args.model_file.endswith(f"{args.seed}_best.th") or args.model_file.endswith(f"{args.seed}_latest.th")
        # check that this model has a split number that matches the dataset
        split = args.eval_split_num
        assert split is not None
        splits = [split]
        match = re.compile(r".*{}_(best|ep-\d+).th$".format(split)).match(args.model_file)
        assert match is not None, args.model_file

    repeat_results = defaultdict(list)

    model_referent_annotation = {}

    num_dots = 7

    for split in splits:
        device_id = utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        domain = get_domain(args.domain)
        if args.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if args.model_dir:
            model_filename = os.path.join(
                args.model_dir,
                f'{split}_{name}.{extension}',
            )
        else:
            model_filename = args.model_file

        try:
            model = utils.load_model(model_filename, map_location=device, prefix_dir=None)
        except FileNotFoundError as e:
            print(e)
            continue
        # model = utils.load_model(args.model_file + '_' + str(seed) + '.th')
        if args.cuda:
            model.cuda()
        model.eval()

        if vars(model.args).get('next_mention_prediction') and vars(model.args).get('next_mention_prediction_type') == 'multi_reference':
            if not args.force_next_mention_num_markables:
                raise NotImplementedError("implement an eval that allows different numbers of gold and predicted mentions")


        corpus = model.corpus_ty(
            domain, args.data,
            train='train_reference_{}.txt'.format(split),
            valid='valid_reference_{}.txt'.format(split),
            test='test_reference_{}.txt'.format(split), #test='selfplay_reference_{}.txt'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True,
            max_instances_per_split=args.max_instances_per_split
        )

        assert model.args.unk_threshold == args.unk_threshold

        with open(os.path.join(args.data, args.transcript_file), "r") as f:
            dialog_corpus = json.load(f)
        with open(os.path.join(args.data, args.markable_file), "r") as f:
            markable_annotation = json.load(f)
        with open(os.path.join(args.data, "aggregated_referent_annotation.json"), "r") as f:
            aggregated_referent_annotation = json.load(f)

        scenarios = {scenario['scenario_uuid']: scenario for scenario in dialog_corpus}

        crit = Criterion(model.word_dict, device_id=device_id, bad_toks = ['<pad>'])
        crit_no_reduce = Criterion(model.word_dict, device_id=device_id, bad_toks = ['<pad>'], reduction='none')
        sel_crit = nn.CrossEntropyLoss()
        ref_crit = nn.BCEWithLogitsLoss()
        ref_crit_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
        is_sel_crit = nn.BCEWithLogitsLoss(reduction='mean')

        if args.eval_split == 'dev':
            testset, testset_stats = corpus.valid_dataset(args.bsz)
        else:
            assert args.eval_split == 'test'
            testset, testset_stats = corpus.test_dataset(args.bsz)
        test_lang_loss, test_select_loss, test_reference_loss, test_select_correct, test_select_total, test_reference_correct, test_reference_total = 0, 0, 0, 0, 0, 0, 0

        """
            Variables to keep track of the results for analysis
        """

        # num_referents --> count, count correct
        total_num_markables = 0
        num_markables_counter = Counter()
        num_markables_correct = Counter()

        exact_match = 0
        exact_match_counter = Counter()

        # location of markable --> count, count correct, count exact match
        location_counter = Counter()
        location_correct = Counter()
        location_exact_match = Counter()

        # information to compute correlation between selection and reference score
        select_correct = {}
        reference_correct = {}
        reference_total = {}

        # markable text --> count, count correct
        text_counter = Counter()
        text_correct = Counter()

        anaphora_list = ["it", "that", "thats", "this", "its", "they", "their", "itself", "them", "those", "it's"]
        total_anaphora = 0
        correct_anaphora = 0

        bleu_scores = []

        merged_args = argparse.Namespace(**utils.merge_dicts(vars(args), vars(model.args)))

        def make_predictor(reference_prediction):
            if reference_prediction == 'l1':
                return PragmaticReferencePredictor(merged_args)
            elif reference_prediction == 'l0':
                return ReferencePredictor(merged_args)
            else:
                raise ValueError(f"invalid --reference_prediction {reference_prediction}")

        reference_predictor = make_predictor(args.reference_prediction)
        partner_reference_predictor = make_predictor(args.partner_reference_prediction)

        if args.next_mention_reranking:
            next_mention_predictor = RerankingMentionPredictor(
                merged_args,
                scoring_function=args.next_mention_reranking_score,
                default_weight=args.next_mention_reranking_default_weight,
                weights=args.next_mention_reranking_weights,
            )
        else:
            next_mention_predictor = ReferencePredictor(merged_args)

        ref_stats = defaultdict(lambda: 0.0)
        partner_ref_stats = defaultdict(lambda: 0.0)
        next_mention_stats = defaultdict(lambda: 0.0)
        is_selection_stats = defaultdict(lambda: 0.0)

        return_all_selection_outs = model.args.is_selection_prediction and args.selection_confidence

        for batch in tqdm.tqdm(testset, ncols=80):
            if isinstance(corpus, ReferenceSentenceCorpus):
                # TODO: add is_selection, and others?
                ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, \
                scenario_ids, real_ids, partner_real_ids, agents, chat_ids, sel_idx, \
                lens, _, _, num_markables, is_self, partner_ref_inpts, partner_ref_tgts_our_view, \
                partner_num_markables, ref_disagreements, partner_ref_disagreements, partner_ref_tgts_their_view, \
                is_selection, non_pronoun_ref_inpts, non_pronoun_ref_tgts, non_pronoun_num_markables, is_augmented = batch
                bsz = ctx.size(0)
                multi_sentence = True
            elif isinstance(corpus, ReferenceCorpus):
                # needs to come second since it's a subclass
                raise NotImplementedError("add extra instance properties")
                ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, \
                scenario_ids, real_ids, partner_real_ids, agents, chat_ids, sel_idx, lens, partner_ref_inpt, partner_ref_tgt_our_view, this_partner_num_markables = batch
                inpts, tgts, ref_inpts, ref_tgts, lens = [inpt], [tgt], [ref_inpt], [ref_tgt], [lens]
                partner_ref_inpts = [partner_ref_inpt]
                partner_ref_tgts_our_view = [partner_ref_tgt_our_view]
                partner_num_markables = [this_partner_num_markables]
                if ref_inpt is None:
                    nm = 0
                else:
                    nm = ref_inpt.size(1)
                num_markables = [
                    torch.full((bsz,), nm).long()
                ]
                multi_sentence = False
            else:
                raise ValueError("invalid corpus type {}".format(type(corpus)))

            ctx = Variable(ctx)
            bsz = ctx.size(0)
            num_dots == int(ctx.size(1) / 4)
            assert num_dots == 7
            inpts = [Variable(inpt) for inpt in inpts]
            ref_inpts = [Variable(ref_inpt) if ref_inpt is not None else None
                         for ref_inpt in ref_inpts]
            tgts = [Variable(tgt) for tgt in tgts]
            sel_tgt = Variable(sel_tgt)

            dots_mentioned_gold = make_dots_mentioned_multi(ref_tgts, model.args, bsz, num_dots)
            dots_mentioned_per_ref_gold = make_dots_mentioned_per_ref_multi(ref_tgts, model.args, bsz, num_dots)
            partner_dots_mentioned_our_view_gold = make_dots_mentioned_multi(
                partner_ref_tgts_our_view, model.args, bsz, num_dots
            )
            partner_dots_mentioned_our_view_per_ref_gold = make_dots_mentioned_per_ref_multi(
                partner_ref_tgts_our_view, model.args, bsz, num_dots
            )

            if isinstance(corpus, ReferenceSentenceCorpus):
                if args.allow_belief_cheating:
                    if model.args.dots_mentioned_no_pronouns:
                        dots_mentioned = make_dots_mentioned_multi(non_pronoun_ref_tgts, model.args, bsz, num_dots)
                        dots_mentioned_per_ref = make_dots_mentioned_per_ref_multi(non_pronoun_ref_tgts, model.args, bsz, num_dots)
                        dots_mentioned_num_markables = non_pronoun_num_markables
                    else:
                        dots_mentioned = dots_mentioned_gold
                        dots_mentioned_per_ref = dots_mentioned_per_ref_gold
                        dots_mentioned_num_markables = num_markables
                    partner_dots_mentioned_our_view = partner_dots_mentioned_our_view_gold
                    partner_dots_mentioned_our_view_per_ref = partner_dots_mentioned_our_view_per_ref_gold

                    belief_constructor = BeliefConstructor(
                        model.args, None,
                        bsz, num_dots, inpts, ref_tgts, partner_ref_tgts_our_view,
                        real_ids, partner_real_ids, sel_tgt, is_self,
                        partner_dots_mentioned_our_view, partner_dots_mentioned_our_view_per_ref,
                        dots_mentioned, dots_mentioned_per_ref,
                        ref_inpts, partner_ref_inpts,
                        num_markables,
                        partner_num_markables,
                    )
                    can_confirm = make_can_confirm(
                        is_self, partner_num_markables, partner_ref_tgts_our_view,
                        resolution_strategy=vars(model.args).get('confirmations_resolution_strategy', 'any')
                    )
                else:
                    # don't cheat!
                    # TODO: feed next mention predictions and previous predictions (for can_confirm); affects ppl
                    dots_mentioned_per_ref = [
                        torch.zeros((bsz, nm.max().item(), 7)).bool().to(device)
                        for nm in num_markables
                    ]
                    dots_mentioned = [torch.zeros((bsz, 7)).bool().to(device) for _ in num_markables]
                    dots_mentioned_num_markables = num_markables
                    belief_constructor = BlankBeliefConstructor()
                    can_confirm = [torch.full((bsz,), CAN_CONFIRM_VALUES[None]).to(device) for _ in num_markables]

                state, outs, ref_outs, sel_outs, ctx_attn_prob, feed_ctx_attn_prob, next_mention_outs, is_selection_outs, \
                (reader_lang_hs, writer_lang_hs), l1_scores, next_mention_rollouts, \
                relation_swapped_ref_outs, relation_swapped_partner_ref_outs, has_relation_swaps = model.forward(
                    ctx, inpts, ref_inpts, sel_idx,
                    num_markables, partner_num_markables,
                    lens,
                    dots_mentioned, dots_mentioned_per_ref, dots_mentioned_num_markables,
                    belief_constructor=belief_constructor,
                    partner_ref_inpts=partner_ref_inpts,
                    ref_tgts=ref_tgts, partner_ref_tgts=partner_ref_tgts_our_view,
                    force_next_mention_num_markables=args.force_next_mention_num_markables,
                    is_selection=is_selection,
                    can_confirm=can_confirm,
                    num_next_mention_candidates_to_score=args.next_mention_reranking_k if args.next_mention_reranking else None,
                    return_all_selection_outs=return_all_selection_outs,
                    relation_swap=False,
                )
            elif isinstance(corpus, ReferenceCorpus):
                # don't cheat!
                assert not args.allow_belief_cheating
                dots_mentioned = None
                selection_beliefs = None
                generation_beliefs = None
                state, out, ref_out, sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_out, (reader_lang_h, writer_lang_h) = model.forward(
                    ctx, inpts[0], ref_inpts[0], sel_idx,
                    num_markables[0], partner_num_markables[0],
                    lens[0], dots_mentioned, belief_constructor,
                    # partner_ref_inpt is used in training if partner reference prediction is supervised
                    partner_ref_inpt=partner_ref_inpts[0],
                    ref_tgts=ref_tgts[0], partner_ref_tgts=partner_ref_tgts_our_view[0],
                    is_selection=is_selection,
                    can_confirm=can_confirm,
                )
                outs, ref_outs = [out], [ref_out]
            else:
                raise ValueError("invalid corpus type {}".format(type(corpus)))

            markables_by_sentence_by_batch = []

            for j in range(bsz):
                chat_id = chat_ids[j]
                remaining_markables = markable_annotation[chat_id]["markables"]

                def keep_markable(markable):
                    markable_id = markable["markable_id"]
                    if markable_id in aggregated_referent_annotation[chat_id] and markable["speaker"] == agents[j]:
                        if "unidentifiable" in aggregated_referent_annotation[chat_id][markable_id] and aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"]:
                            return False
                        return True
                    return False

                if not multi_sentence:
                    markables = [
                        markable for markable in remaining_markables
                        if keep_markable(markable)
                    ]
                    markables_by_sentence = [markables]
                else:
                    markables_by_sentence = []
                    # markables information from aggregated_referent_annotation

                    chat = next(chat for chat in dialog_corpus if chat['uuid'] == chat_id)
                    messages = markable_annotation[chat_id]['text'].split('\n')

                    assert len(messages) == len(inpts)

                    # def foo(ix):
                    #     start = m[ix]['start'] - 3 * (ix + 1)
                    #     end   = m[ix]['end'] - 3 * (ix + 1)
                    #     start -= sum(len(m) for m in messages[:ix])
                    #     end   -= sum(len(m) for m in messages[:ix])
                    #     return messages[ix][start:end]

                    acc_sent_length = 0
                    for sentence_ix, _ in enumerate(inpts):
                        this_len = len(messages[sentence_ix]) + 1 # add 1 for newline
                        markables = []
                        if not remaining_markables:
                            markables_by_sentence.append(markables)
                            continue
                        while remaining_markables and remaining_markables[0]['end'] < acc_sent_length + this_len:
                            markable, remaining_markables = remaining_markables[0], remaining_markables[1:]
                            if keep_markable(markable):
                                markables.append(markable)

                                extracted_text = messages[sentence_ix][markable['start']-acc_sent_length:markable['end']-acc_sent_length]
                                if markable['text'] != extracted_text:
                                    print(chat_id)
                                    print(j, sentence_ix, markable['text'])
                                    print(j, sentence_ix, extracted_text)
                                    print(acc_sent_length)
                                    print(markable)
                                    assert False
                        acc_sent_length += this_len
                        markables_by_sentence.append(markables)

                    assert not remaining_markables
                markables_by_sentence_by_batch.append(markables_by_sentence)

            my_ref_outs, partner_ref_outs = zip(*ref_outs)

            # next_mention_outs[i]: dots predicted to be mentioned in sentence i (TODO: rename this)
            for sentence_ix, (inpt, out, tgt, ref_inpt, partner_ref_inpt, (ref_out, partner_ref_out), ref_tgt, partner_ref_tgt, next_mention_out, this_is_selection, is_selection_out) in enumerate(
                utils.safe_zip(inpts, outs, tgts, ref_inpts, partner_ref_inpts, ref_outs, ref_tgts, partner_ref_tgts_our_view, next_mention_outs[:-1], is_selection, is_selection_outs)
            ):
                sentence_num_markables = num_markables[sentence_ix]
                tgt = Variable(tgt)
                # lang_loss = crit(out, tgt)

                lang_loss = crit_no_reduce(out, tgt).view(-1, bsz)
                if args.lang_only_self:
                    assert multi_sentence
                    lang_loss = lang_loss * is_self[sentence_ix].unsqueeze(0).expand_as(lang_loss)
                lang_loss = lang_loss.sum()

                if sentence_ix == 0:
                    assert reader_lang_hs[sentence_ix] is None and writer_lang_hs[sentence_ix] is None
                    reader_and_writer_lang_h = None
                else:
                    reader_and_writer_lang_h = (reader_lang_hs[sentence_ix], writer_lang_hs[sentence_ix])
                if ref_inpt is not None:
                    if args.reference_prediction == 'l1':
                        scoring_function = model.make_l1_scoring_function(
                            state, inpt, tgt, ref_inpt,
                            num_markables[sentence_ix], partner_num_markables[sentence_ix],
                            lens[sentence_ix], reader_and_writer_lang_h,
                            belief_constructor=belief_constructor, partner_ref_inpt=partner_ref_inpts[sentence_ix],
                            timestep=sentence_ix, partner_ref_outs=partner_ref_outs, ref_outs=my_ref_outs,
                            next_mention_out=next_mention_out
                        )
                        ref_loss, ref_predictions, _ref_stats = reference_predictor.forward(
                            ref_inpt, ref_tgt, ref_out, sentence_num_markables, scoring_function,
                            by_num_markables=True,
                        )
                    else:
                        ref_loss, ref_predictions, _ref_stats = reference_predictor.forward(
                            ref_inpt, ref_tgt, ref_out, sentence_num_markables,
                            by_num_markables=True,
                        )
                    ref_stats = utils.sum_dicts(ref_stats, _ref_stats)

                    ref_tgt = ref_tgt.transpose(0,1).contiguous()

                    # compute more details of reference resolution
                    if ref_predictions is not None:
                        for j in range(bsz): # batch idx
                            chat_id = chat_ids[j]
                            markables = markables_by_sentence_by_batch[j][sentence_ix]
                            this_num_markables = sentence_num_markables[j].item()
                            assert len(markables) == this_num_markables

                            # add chat level details if not exists
                            # TODO: fix this
                            if chat_id not in reference_correct:
                                reference_correct[chat_id] = _ref_stats['correct']
                            if chat_id not in reference_total:
                                reference_total[chat_id] = _ref_stats['num_dots']
                            if chat_id not in model_referent_annotation:
                                model_referent_annotation[chat_id] = {}

                            for i in range(this_num_markables): # markable idx
                                # fixed a typo (?) here that was autobroadcasting, but it shouldn't affect correctness
                                correct_result = (ref_predictions.long() == ref_tgt.long())[i,j].sum().item()
                                exact_match_result = torch.equal(ref_predictions.long()[i][j], ref_tgt.long()[i][j])

                                """
                                    Add information to variables
                                """
                                total_num_markables += 1
                                num_markables_counter[ref_tgt.long()[i][j].sum().item()] += 1
                                num_markables_correct[ref_tgt.long()[i][j].sum().item()] += correct_result

                                # compute exact match
                                if exact_match_result:
                                    exact_match += 1
                                    exact_match_counter[ref_tgt.long()[i][j].sum().item()] += 1
                                    text_correct[markables[i]["text"].lower()] += 1

                                location_correct[i] += correct_result
                                if exact_match_result:
                                    location_exact_match[i] += 1
                                location_counter[i] += 1

                                text_counter[markables[i]["text"].lower()] += 1

                                # test anaphora
                                if markables[i]["text"].lower() in anaphora_list:
                                    total_anaphora += 1
                                    if exact_match_result:
                                        correct_anaphora += 1

                                # keep track of model predictions for later visualization
                                chat = [chat for chat in dialog_corpus if chat['uuid'] == chat_id]
                                chat = chat[0]
                                if markables[i]['markable_id'] not in model_referent_annotation[chat_id]:
                                    model_referent_annotation[chat_id][markables[i]['markable_id']] = {}
                                    model_referent_annotation[chat_id][markables[i]['markable_id']]['referents'] = []
                                    model_referent_annotation[chat_id][markables[i]['markable_id']]['ambiguous'] = False
                                    model_referent_annotation[chat_id][markables[i]['markable_id']]['unidentifiable'] = False
                                    for ent, is_referent in zip(chat['scenario']['kbs'][agents[j]], ref_predictions.long()[i][j].tolist()):
                                        if is_referent:
                                            model_referent_annotation[chat_id][markables[i]['markable_id']]['referents'].append("agent_{}_{}".format(agents[j], ent['id']))

                    # moved these here to fix a bug; previously some batches were getting double-counted if the batch afterward had no referents
                    test_reference_correct += _ref_stats['correct']
                    test_reference_total += _ref_stats['num_dots']
                else:
                    ref_loss = None

                if partner_ref_inpt is not None:
                    if args.partner_reference_prediction == 'l1':
                        if not args.l1_oracle:
                            raise NotImplementedError("non-oracle l1 for partner reference")
                        partner_ref_loss, partner_ref_predictions, _partner_ref_stats = partner_reference_predictor.forward(
                            partner_ref_inpt, partner_ref_tgt, partner_ref_out, partner_num_markables[sentence_ix], None,
                            by_num_markables=True
                        )
                    else:
                        partner_ref_loss, partner_ref_predictions, _partner_ref_stats = partner_reference_predictor.forward(
                            partner_ref_inpt, partner_ref_tgt, partner_ref_out, partner_num_markables[sentence_ix],
                            by_num_markables=True,
                        )

                    partner_ref_stats = utils.sum_dicts(partner_ref_stats, _partner_ref_stats)

                test_lang_loss += lang_loss.item()
                if ref_loss:
                    test_reference_loss += ref_loss.item()

                if model.args.next_mention_prediction:
                    pred_dots_mentioned, next_mention_stop_loss, pred_next_mention_num_markables = next_mention_out
                    if model.args.next_mention_prediction_type == 'multi_reference':
                        gold_num_mentions = num_markables[sentence_ix].long()
                        gold_dots_mentioned = dots_mentioned_per_ref_gold[sentence_ix].long()
                        # next_mention_stop_losses.append(next_mention_stop_loss)
                        expanded_next_mention_loss = True
                    elif model.args.next_mention_prediction_type == 'collapsed':
                        # supervise only for self, so pseudo_num_mentions = 1 iff is_self; 0 otherwise
                        gold_num_mentions = is_self[sentence_ix].long()
                        gold_dots_mentioned = dots_mentioned_gold[sentence_ix].long().unsqueeze(1)
                        expanded_next_mention_loss = False
                    else:
                        raise ValueError(f"--next_mention_prediction_type={self.args.next_mention_prediction}")
                    if args.next_mention_reranking:
                        # hack; pass True for inpt because this method only uses it to ensure it's not null
                        num_mentions_per_candidate = gold_num_mentions.unsqueeze(1).expand(
                            -1, self.args.next_mention_reranking_k
                        )
                        _loss, _pred, _stats = next_mention_predictor.forward(
                            True, gold_dots_mentioned, pred_dots_mentioned, num_mentions_per_candidate,
                            next_mention_rollouts[sentence_ix],
                            # TODO: collapse param naming is misleading; collapse adds in additional expanded_* terms
                            collapse=expanded_next_mention_loss,
                        )
                    else:
                        # hack; pass True for inpt because this method only uses it to ensure it's not null
                        _loss, _pred, _stats = next_mention_predictor.forward(
                            True, gold_dots_mentioned, pred_dots_mentioned, gold_num_mentions,
                            # TODO: collapse param naming is misleading; collapse adds in additional expanded_* terms
                            collapse=expanded_next_mention_loss,
                        )
                    next_mention_stats = utils.sum_dicts(next_mention_stats, _stats)

                if model.args.is_selection_prediction:
                    if args.selection_confidence and (sentence_ix > 0 or args.selection_confidence_look_ahead):
                        ix_to_use = sentence_ix if args.selection_confidence_look_ahead else sentence_ix - 1
                        max_selection_log_probs = sel_outs[ix_to_use][0].log_softmax(-1).max(-1).values
                        combined_scores = max_selection_log_probs * args.selection_confidence_weight + is_selection_out.sigmoid().log() * (1 - args.selection_confidence_weight)
                        scores_to_choose = combined_scores - torch.tensor(args.selection_confidence_threshold).log()
                    else:
                        scores_to_choose = is_selection_out

                    add_selection_stats(this_is_selection, scores_to_choose, is_selection_stats)

                # END loop over sentences

            if return_all_selection_outs:
                sel_logits, _, _ = sel_outs[-1]
            else:
                sel_logits, _, _ = sel_outs

            sel_loss = sel_crit(sel_logits, sel_tgt)
            sel_correct = (sel_logits.max(dim=1)[1] == sel_tgt).sum().item()
            sel_total = sel_logits.size(0)
            for i in range(sel_tgt.size(0)): # batch idx
                chat_id = chat_ids[i]
                sel_resuts = (sel_logits.max(dim=1)[1] == sel_tgt)
                if sel_resuts[i]:
                    select_correct[chat_id] = 1
                else:
                    select_correct[chat_id] = 0
            test_select_loss += sel_loss.item()
            test_select_correct += sel_correct
            test_select_total += sel_total

            if args.bleu_n > 0:
                ctx_h = model.ctx_encoder(ctx.transpose(0,1))

                my_utterance = None
                idx = 0
                while True:
                    if inpt[idx] == model.word_dict.word2idx['YOU:']:
                        start = idx
                        my_utterance = model.read_and_write(
                            inpt[:idx], ctx_h, 30, temperature=args.temperature)
                        my_utterance = model.word_dict.i2w(my_utterance)
                        #print(my_utterance)
                        while not inpt[idx] in [model.word_dict.word2idx[stop_token] for stop_token in data.STOP_TOKENS]:
                            idx += 1
                        end = idx
                        golden_utterance = inpt[start:end]
                        golden_utterance = model.word_dict.i2w(golden_utterance)
                        bleu_scores.append(100 * sentence_bleu([golden_utterance], my_utterance, weights=[1 for i in range(4) if args.bleu_n == i], #weights=[1 / args.bleu_n] * args.bleu_n,
                                                               smoothing_function=SmoothingFunction().method7))
                    if inpt[idx] == model.word_dict.word2idx['<selection>']:
                        break

                    idx += 1

        metrics = {}
        flat_stats = flatten_metrics({
            'ref_stats': ref_stats,
            'partner_ref_stats': partner_ref_stats,
            'next_mention_stats': next_mention_stats,
            'is_selection_stats': is_selection_stats,
        })
        pprint.pprint(flat_stats)
        add_metrics(flat_stats, metrics, 'ref')
        if model.args.partner_reference_prediction:
            add_metrics(flat_stats, metrics, 'partner_ref')
        if args.l1_speaker_weights:
            for weight in args.l1_speaker_weights:
                add_metrics(flat_stats, metrics, f'ref_sw-{weight:.2f}')

        if model.args.next_mention_prediction:
            add_metrics(flat_stats, metrics, "next_mention")
            if model.args.next_mention_prediction_type == 'multi_reference':
                add_metrics(flat_stats, metrics, "next_mention_expanded")
            if args.next_mention_reranking:
                for weight in args.next_mention_reranking_weights:
                    add_metrics(flat_stats, metrics, f'next_mention_sw-{weight:.2f}')
                    add_metrics(flat_stats, metrics, f'next_mention_sw-{weight:.2f}_expanded')

        if model.args.is_selection_prediction:
            add_metrics(flat_stats, metrics, "is_selection", compute_em=False, compute_baseline=True)

        for num_markables in [1]:
            add_metrics(flat_stats, metrics, f'ref_nm-{num_markables}')
            if model.args.partner_reference_prediction:
                add_metrics(flat_stats, metrics, f'partner_ref_nm-{num_markables}')
            if args.l1_speaker_weights:
                for weight in args.l1_speaker_weights:
                    add_metrics(flat_stats, metrics, f'ref_sw-{weight:.2f}_nm-{num_markables}')
        pprint.pprint(metrics)

        # Main results:
        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        if args.lang_only_self:
            test_lang_loss /= testset_stats['self_nonpadn']
            lang_loss_name = 'langloss[SELF]'
            lang_ppl_name = 'langppl[SELF]'
        else:
            test_lang_loss /= testset_stats['nonpadn']
            lang_loss_name = 'langloss'
            lang_ppl_name = 'langppl'
        test_select_loss /= len(testset)
        test_select_accuracy = test_select_correct / test_select_total
        metrics['select_acc'] = test_select_accuracy
        test_reference_accuracy = test_reference_correct / test_reference_total
        print('eval_reference_correct {} ; eval_reference_total {}'.format(test_reference_correct, test_reference_total))
        # this is affected by --allow_belief_cheating for models that constrain attention or mark dots mentioned
        # print('eval%s %.8f | eval%s %.8f' % (lang_loss_name, test_lang_loss, lang_ppl_name, np.exp(test_lang_loss)))
        print('evalselectloss %.8f | evalselectaccuracy %.6f' % (test_select_loss, test_select_accuracy))
        print('evalreferenceloss %.8f | evalreferenceaccuracy %.6f' % (test_reference_loss, test_reference_accuracy))
        print('reference_exact_match %.6f' % (exact_match / total_num_markables))
        for k in sorted(num_markables_counter.keys()):
            print('{}: {:.4f} {:.4f} (out of {})'.format(k, num_markables_correct[k] / (num_markables_counter[k] * 7), exact_match_counter[k] / num_markables_counter[k], num_markables_counter[k]))
        print('eval anaphora: {} (out of {})'.format(correct_anaphora / total_anaphora, total_anaphora))

        print("print string:")
        default_stat_values = [metrics.get(stat_name, 0.0) for stat_name in DEFAULT_STATS]
        print(','.join(DEFAULT_STATS))
        print(','.join('{:.4f}'.format(x) for x in default_stat_values))

        if args.bleu_n > 0:
            print('average bleu score {}'.format(np.mean(bleu_scores)))

        # reference/selection correlation
        reference_score = []
        selection_score = []
        for chat_id in sorted(reference_correct.keys()):
            reference_score.append(reference_correct[chat_id] / reference_total[chat_id])
            selection_score.append(select_correct[chat_id])
        plt.xlabel('reference score', fontsize=14)
        plt.ylabel('selection score', fontsize=14)
        #ax = sns.scatterplot(x=reference_score, y=selection_score, size=0, legend=False)
        sns.regplot(x=reference_score, y=selection_score)
        #plt.axes().set_aspect('equal', 'datalim')
        plt.savefig('reference_selection_{}.png'.format(split), dpi=300)
        plt.clf()
        reference_score = np.array(reference_score)
        selection_score = np.array(selection_score)
        print("reference selection correlation: {}".format(np.corrcoef(reference_score, selection_score)))

        # keep track of results for this run
        repeat_results["eval_lang_loss"].append(test_lang_loss)
        repeat_results["eval_select_loss"].append(test_select_loss)
        repeat_results["eval_select_accuracy"].append(test_select_accuracy)
        repeat_results["eval_reference_loss"].append(test_reference_loss)
        repeat_results["eval_reference_accuracy"].append(test_reference_accuracy)
        repeat_results["correlation_score"].append(np.corrcoef(reference_score, selection_score)[0][1])
        repeat_results["num_markables_counter"].append(copy.copy(num_markables_counter))
        repeat_results["exact_match_counter"].append(copy.copy(exact_match_counter))
        repeat_results["num_markables_correct"].append(copy.copy(num_markables_correct))
        repeat_results["reference_exact_match"].append(exact_match / total_num_markables)
        repeat_results["test_perplexity"].append(np.exp(test_lang_loss))
        repeat_results["location_counter"].append(copy.copy(location_counter))
        repeat_results["location_correct"].append(copy.copy(location_correct))
        repeat_results["location_exact_match"].append(copy.copy(location_exact_match))

    if args.lang_only_self:
        lang_loss_name = 'loss [SELF]'
        lang_ppl_name = 'perplexity [SELF]'
    else:
        test_lang_loss /= testset_stats['nonpadn']
        lang_loss_name = 'loss'
        lang_ppl_name = 'perplexity'

    number_models_averaged = len(repeat_results['eval_lang_loss'])

    if number_models_averaged > 1:

        print("=================================\n\n")
        print("number of models averaged: {}".format(number_models_averaged))
        print("repeat eval lang %s %.8f" % (lang_loss_name, np.mean(repeat_results["eval_lang_loss"])))
        print("repeat eval select loss %.8f" % np.mean(repeat_results["eval_select_loss"]))
        print("repeat eval select accuracy %.8f ( %.8f )" % (np.mean(repeat_results["eval_select_accuracy"]), np.std(repeat_results["eval_select_accuracy"])))
        print("repeat eval reference loss %.8f" % np.mean(repeat_results["eval_reference_loss"]))
        print("repeat eval reference accuracy %.8f ( %.8f )" % (np.mean(repeat_results["eval_reference_accuracy"]), np.std(repeat_results["eval_reference_accuracy"])))
        print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
        print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
        print("repeat reference exact match %.8f ( %.8f )" % (np.mean(repeat_results["reference_exact_match"]), np.std(repeat_results["reference_exact_match"])))
        print("repeat eval %s %.8f ( %.8f )" % (lang_ppl_name, np.mean(repeat_results["eval_perplexity"]), np.std(repeat_results["eval_perplexity"])))

        for k in sorted(num_markables_counter.keys()):
            print("repeat accuracy and exact match:")
            num_markables = []
            exact_match = []
            exact_match_rate = []
            num_markables_correct = []
            for split in range(len(splits)):
                if split >= len(repeat_results["num_markables_counter"]):
                    continue
                num_markables.append(repeat_results["num_markables_counter"][split][k])
                exact_match.append(repeat_results["exact_match_counter"][split][k])
                exact_match_rate.append(repeat_results["exact_match_counter"][split][k] / repeat_results["num_markables_counter"][split][k])
                num_markables_correct.append(repeat_results["num_markables_correct"][split][k] / (repeat_results["num_markables_counter"][split][k] * 7))
            print('{}: {:.5f} (std {}) {:.5f} (std {}) (count {})'.format(k, np.mean(num_markables_correct), np.std(num_markables_correct), np.mean(exact_match_rate), np.std(exact_match_rate), np.mean(num_markables)))

    if args.model_referent_annotation_output_path:
        dump_json(model_referent_annotation, args.model_referent_annotation_output_path)

    # pdb.set_trace()

if __name__ == '__main__':
    with torch.no_grad():
        main()
