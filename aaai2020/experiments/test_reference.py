"""
Performs evaluation of the model on the test dataset.
"""

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
import pprint
import sys

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


def main():
    parser = argparse.ArgumentParser(description='testing script for reference resolution')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--model_file', type=str, required=True,
        help='pretrained model file')
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

    # for error analysis
    parser.add_argument('--transcript_file', type=str, default='final_transcripts.json',
        help='scenario file')
    parser.add_argument('--markable_file', type=str, default='markable_annotation.json',
        help='scenario file')
    parser.add_argument('--show_errors', action='store_true', default=False,
        help='show errors')

    args = parser.parse_args()

    utils.dump_git_status(sys.stdout)
    print(' '.join(sys.argv))
    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.bleu_n > 0:
        # current support
        args.bsz = 1

    if args.repeat_test:
        seeds = list(range(10))
    else:
        seeds = [args.seed]

    repeat_results = defaultdict(list)

    model_referent_annotation = {}

    for seed in seeds:
        device_id = utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        def model_filename_fn(name, extension):
            return '{}_{}_{}.{}'.format(args.model_file, seed, name, extension)

        domain = get_domain(args.domain)
        if args.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        try:
            model = utils.load_model(model_filename_fn('best', 'th'), map_location=device)
        except FileNotFoundError as e:
            print(e)
            continue
        # model = utils.load_model(args.model_file + '_' + str(seed) + '.th')
        if args.cuda:
            model.cuda()
        model.eval()

        corpus = model.corpus_ty(domain, args.data, train='train_reference_{}.txt'.format(seed), valid='valid_reference_{}.txt'.format(seed), test='test_reference_{}.txt'.format(seed), #test='selfplay_reference_{}.txt'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True)

        with open(os.path.join(args.data, args.transcript_file), "r") as f:
            dialog_corpus = json.load(f)
        with open(os.path.join(args.data, args.markable_file), "r") as f:
            markable_annotation = json.load(f)
        with open(os.path.join(args.data, "aggregated_referent_annotation.json"), "r") as f:
            aggregated_referent_annotation = json.load(f)

        scenarios = {scenario['scenario_uuid']: scenario for scenario in dialog_corpus}

        crit = Criterion(model.word_dict, device_id=device_id)
        sel_crit = nn.CrossEntropyLoss()
        ref_crit = nn.BCEWithLogitsLoss()
        ref_crit_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

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

        for batch in testset:
            if isinstance(corpus, ReferenceSentenceCorpus):
                ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, scenario_ids, real_ids, agents, chat_ids, sel_idx, lens, _, _, num_markables_by_sentence = batch
                bsz = ctx.size(0)
            elif isinstance(corpus, ReferenceCorpus):
                # needs to come second since it's a subclass
                ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, real_ids, agents, chat_ids, sel_idx = batch
                bsz = ctx.size(0)
                inpts, tgts, ref_inpts, ref_tgts = [inpt], [tgt], [ref_inpt], [ref_tgt]
                lens = None
                if ref_inpt is None:
                    nm = 0
                else:
                    nm = ref_inpt.size(1)
                num_markables_by_sentence = [
                    torch.full((bsz,), nm).long()
                ]
            else:
                raise ValueError("invalid corpus type {}".format(type(corpus)))

            ctx = Variable(ctx)
            inpts = [Variable(inpt) for inpt in inpts]
            ref_inpts = [Variable(ref_inpt) if ref_inpt is not None else None
                         for ref_inpt in ref_inpts]
            tgts = [Variable(tgt) for tgt in tgts]
            sel_tgt = Variable(sel_tgt)


            if isinstance(corpus, ReferenceSentenceCorpus):
                # don't cheat!
                dots_mentioned = [None] * len(inpts)
                outs, ref_outs, sel_out, ctx_attn_prob, feed_ctx_attn_prob = model.forward(
                    ctx, inpts, ref_inpts, sel_idx, lens, dots_mentioned
                )
            elif isinstance(corpus, ReferenceCorpus):
                # don't cheat!
                dots_mentioned = None
                out, ref_out, sel_out, ctx_attn_prob, feed_ctx_attn_prob = model.forward(
                    ctx, inpt, ref_inpt, sel_idx, lens, dots_mentioned
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

                if lens is None:
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

            for sentence_ix, (inpt, out, tgt, ref_inpt, ref_out, ref_tgt) in enumerate(
                utils.safe_zip(inpts, outs, tgts, ref_inpts, ref_outs, ref_tgts)
            ):
                sentence_num_markables = num_markables_by_sentence[sentence_ix]
                tgt = Variable(tgt)
                lang_loss = crit(out, tgt)

                if ref_inpt is not None:
                    ref_tgt = Variable(ref_tgt)
                    ref_tgt = torch.transpose(ref_tgt, 0, 1).contiguous().float()
                    ref_mask = torch.zeros_like(ref_tgt)
                    for i, nm in enumerate(sentence_num_markables):
                        ref_mask[:nm, i, :] = 1

                    ref_loss = (ref_crit_no_reduce(ref_out, ref_tgt) * ref_mask.float()).sum()
                    ref_correct = (((ref_out > 0).long() == ref_tgt.long()) * ref_mask.byte()).sum().item()
                    ref_total = ref_mask.sum().item()
                    # ref_loss = ref_crit(ref_out, ref_tgt)
                    # t = Variable(torch.FloatTensor([0])) # threshold
                    ref_results = ((ref_out > 0).long() == ref_tgt.long()) * ref_mask
                    # ref_correct = ((ref_out > 0).long() == ref_tgt.long()).sum().item()
                    # ref_total = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)

                    # compute more details of reference resolution
                    for j in range(bsz): # batch idx
                        markables = markables_by_sentence_by_batch[j][sentence_ix]
                        this_num_markables = sentence_num_markables[j].item()
                        assert len(markables) == this_num_markables

                        # add chat level details if not exists
                        if chat_id not in reference_correct:
                            reference_correct[chat_id] = ref_results[:,j,:].sum().item()
                        if chat_id not in reference_total:
                            reference_total[chat_id] = ref_mask[:,j,:].sum() #ref_results[:,j,:].size(0) * ref_results[:,j,:].size(1)
                        if chat_id not in model_referent_annotation:
                            model_referent_annotation[chat_id] = {}

                        for i in range(this_num_markables): # markable idx
                            # fixed a typo (?) here that was autobroadcasting, but it shouldn't affect correctness
                            correct_result = ((ref_out > 0).long() == ref_tgt.long())[i,j].sum().item()
                            exact_match_result = torch.equal((ref_out > 0).long()[i][j], ref_tgt.long()[i][j])

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
                                for ent, is_referent in zip(chat['scenario']['kbs'][agents[j]], (ref_out > 0).long()[i][j].tolist()):
                                    if is_referent:
                                        model_referent_annotation[chat_id][markables[i]['markable_id']]['referents'].append("agent_{}_{}".format(agents[j], ent['id']))
                else:
                    ref_loss = None
                    ref_correct = 0
                    ref_total = 0

                test_lang_loss += lang_loss.item()
                if ref_loss:
                    test_reference_loss += ref_loss.item()
                test_reference_correct += ref_correct
                test_reference_total += ref_total

                # END loop over sentences

            sel_loss = sel_crit(sel_out, sel_tgt)
            sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
            sel_total = sel_out.size(0)
            for i in range(sel_tgt.size(0)): # batch idx
                chat_id = chat_ids[i]
                sel_resuts = (sel_out.max(dim=1)[1] == sel_tgt)
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

        # Main results:
        # Dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        test_lang_loss /= testset_stats['nonpadn']
        test_select_loss /= len(testset)
        test_select_accuracy = test_select_correct / test_select_total
        test_reference_accuracy = test_reference_correct / test_reference_total
        print('test_reference_correct {} ; test_reference_total {}'.format(test_reference_correct, test_reference_total))
        print('testlangloss %.8f | testlangppl %.8f' % (test_lang_loss, np.exp(test_lang_loss)))
        print('testselectloss %.8f | testselectaccuracy %.6f' % (test_select_loss, test_select_accuracy))
        print('testreferenceloss %.8f | testreferenceaccuracy %.6f' % (test_reference_loss, test_reference_accuracy))
        print('reference_exact_match %.6f' % (exact_match / total_num_markables))
        for k in sorted(num_markables_counter.keys()):
            print('{}: {:.4f} {:.4f} (out of {})'.format(k, num_markables_correct[k] / (num_markables_counter[k] * 7), exact_match_counter[k] / num_markables_counter[k], num_markables_counter[k]))
        print('test anaphora: {} (out of {})'.format(correct_anaphora / total_anaphora, total_anaphora))

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
        plt.savefig('reference_selection_{}.png'.format(seed), dpi=300)
        plt.clf()
        reference_score = np.array(reference_score)
        selection_score = np.array(selection_score)
        print("reference selection correlation: {}".format(np.corrcoef(reference_score, selection_score)))

        # keep track of results for this run
        repeat_results["test_lang_loss"].append(test_lang_loss)
        repeat_results["test_select_loss"].append(test_select_loss)
        repeat_results["test_select_accuracy"].append(test_select_accuracy)
        repeat_results["test_reference_loss"].append(test_reference_loss)
        repeat_results["test_reference_accuracy"].append(test_reference_accuracy)
        repeat_results["correlation_score"].append(np.corrcoef(reference_score, selection_score)[0][1])
        repeat_results["num_markables_counter"].append(copy.copy(num_markables_counter))
        repeat_results["exact_match_counter"].append(copy.copy(exact_match_counter))
        repeat_results["num_markables_correct"].append(copy.copy(num_markables_correct))
        repeat_results["reference_exact_match"].append(exact_match / total_num_markables)
        repeat_results["test_perplexity"].append(np.exp(test_lang_loss))
        repeat_results["location_counter"].append(copy.copy(location_counter))
        repeat_results["location_correct"].append(copy.copy(location_correct))
        repeat_results["location_exact_match"].append(copy.copy(location_exact_match))




    print("=================================\n\n")
    print("number of models averaged: {}".format(len(repeat_results['test_lang_loss'])))
    print("repeat test lang loss %.8f" % np.mean(repeat_results["test_lang_loss"]))
    print("repeat test select loss %.8f" % np.mean(repeat_results["test_select_loss"]))
    print("repeat test select accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_select_accuracy"]), np.std(repeat_results["test_select_accuracy"])))
    print("repeat test reference loss %.8f" % np.mean(repeat_results["test_reference_loss"]))
    print("repeat test reference accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_reference_accuracy"]), np.std(repeat_results["test_reference_accuracy"])))
    print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
    print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
    print("repeat reference exact match %.8f ( %.8f )" % (np.mean(repeat_results["reference_exact_match"]), np.std(repeat_results["reference_exact_match"])))
    print("repeat test perplexity %.8f ( %.8f )" % (np.mean(repeat_results["test_perplexity"]), np.std(repeat_results["test_perplexity"])))

    for k in sorted(num_markables_counter.keys()):
        print("repeat accuracy and exact match:")
        num_markables = []
        exact_match = []
        exact_match_rate = []
        num_markables_correct = []
        for seed in range(len(seeds)):
            if seed >= len(repeat_results["num_markables_counter"]):
                continue
            num_markables.append(repeat_results["num_markables_counter"][seed][k])
            exact_match.append(repeat_results["exact_match_counter"][seed][k])
            exact_match_rate.append(repeat_results["exact_match_counter"][seed][k] / repeat_results["num_markables_counter"][seed][k])
            num_markables_correct.append(repeat_results["num_markables_correct"][seed][k] / (repeat_results["num_markables_counter"][seed][k] * 7))
        print('{}: {:.5f} (std {}) {:.5f} (std {}) (count {})'.format(k, np.mean(num_markables_correct), np.std(num_markables_correct), np.mean(exact_match_rate), np.std(exact_match_rate), np.mean(num_markables)))

    dump_json(model_referent_annotation, "model_referent_annotation.json")

    # pdb.set_trace()

if __name__ == '__main__':
    main()
