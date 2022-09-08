import sys
import json

import time

from pathlib import Path

import numpy as np

import torch

from nltk import word_tokenize

from engines.beliefs import BlankBeliefConstructor
from metric import MetricsContainer
from corpora import data
import domain
from dialog import DialogLogger, HierarchicalDialog

from agent import GenerationOutput
from belief_agent import BeliefAgent
from belief import (
    AndOrBelief, OrAndBelief, OrBelief, OrAndOrBelief,
    ConfigBelief,
    process_ctx, expand_plan,
    Label, label_config_sets,
)

from cog_belief import CostBelief, EgoCostBelief

"""
Dialog: [Turn]
Turn: [Utterance, Response, [PriorNextMention], [PlanNextMention]]
PriorNextMention: Set[Dots]
PlanNextMention: Set[Dots]
"""

def undo_state_writer(writer):
    # line numbers might be a little off
    # rnn state (agent.py:866,899)
    # feed in do_update=False to write, no list to pop
    # lang hs (agent.py)
    writer.dots_mentioned_per_ref_chosen.pop() # agent.py:858
    #writer.logprobs.pop() # agent.py:861, added do_update guard
    writer.reader_lang_hs.pop() # agent.py:863
    writer.writer_lang_hs.pop() # agent.py:864
    # ref pred (agent.py:883)
    writer.ref_outs.pop() # agent.py:290
    writer.ref_preds.pop() # agent.py:296
    # these two should be None
    writer.partner_ref_outs.pop() # agent.py:888
    writer.partner_ref_preds.pop() # agent.py:889
    writer.words.pop() # agent.py:891
    writer.sents.pop() # agent.py:892
    # extras
    writer.extras.pop() # agent.py:893
    if len(writer.joint_scores) > 0:
        writer.joint_scores.pop() # agent.py:896
        writer.ref_resolution_scores.pop() # agent.py:897
        writer.language_model_scores.pop() # agent.py:898
    # next mentions (agent.py:636,642)
    if len(writer.next_mention_latents) > 0:
        writer.next_mention_latents.pop() # agent.py:955,974
        writer.next_mention_candidates.pop() # agent.py:957,976
        writer.next_mention_predictions.pop() # agent.py:958,977
        writer.next_mention_predictions_multi.pop() # agent.py:959,978
        writer.next_mention_indices_sorted.pop() # agent.py:960,979
    # markables (agent.py:878-881)
    writer.ref_inpts.pop()
    writer.markables.pop()
    writer.partner_ref_inpts.pop()
    writer.partner_markables.pop()
    # selection (agent.py:919)
    #writer.is_selection_outs.pop() # agent.py:983
    # ^ gated the above behind do_update

def undo_state_reader(reader):
    # line numbers might be off
    reader.sents.pop() # agent.py:365
    reader.reader_lang_hs.pop() # agent.py:381
    reader.writer_lang_hs.pop() # agent.py:382

    reader.ref_inpts.pop() # agent.py:393
    reader.markables.pop() # agent.py:394
    reader.partner_ref_inpts.pop() # agent.py:395
    reader.partner_markables.pop() # agent.py:396

    reader.partner_ref_outs.pop() # agent.py:469
    reader.partner_ref_preds.pop() # agent.py:470

    reader.ref_outs.pop() # agent.py:471
    reader.ref_preds.pop() # agent.py:472
    reader.words.pop() # agent.py:473
    reader.words.pop() # agent.py:474


class StaticDialogLogger:
    def __init__(self, scenario_id, dir="analysis_log"):
        self.filepath = (Path(dir) / scenario_id).with_suffix(".json")
        self.dialogue = []

    def dump_json(self):
        with self.filepath.open("w") as f:
            json.dump(self.dialogue, f, indent=4, sort_keys=True)

    def start_turn(self, writer_id, reader_id):
        self.turn = {
            "writer_id": writer_id,
            "reader_id": reader_id,
        }
    def end_turn(self):
        self.dialogue.append(self.turn)

    def add_turn_utt(
        self,
        configs,
        utterance_language = None,
        utterance = None,
        prior_mentions = None,
        plan_mentions = None,
        # ROUNTDRIP
        prior_mentions_language = None,
        plan_mentions_language = None,
        prior_plan = None,
        plan_plan = None,
        prior_partner_ref = None,
        plan_partner_ref = None,
        # BEAM
        prior_beam_sents = None,
        prior_beam_ref_res = None,
        prior_beam_lm = None,
        plan_beam_sents = None,
        plan_beam_ref_res = None,
        plan_beam_lm = None,
        plan_beam_seed = None,
        # PLAN FEATURE LABEL,
        plan_prior = None,
        writer_configs_prior = None,
        reader_configs_prior = None,
        label_prior = None,
        plan = None,
        writer_configs = None,
        reader_configs = None,
        label = None,
    ):
        self.turn["all_configs"] = configs
        self.turn["utterance_language"] = utterance_language
        self.turn["utterance"] = utterance
        self.turn["prior_mentions"] = prior_mentions
        self.turn["plan_mentions"] = plan_mentions
        self.turn["prior_mentions_language"] = prior_mentions_language 
        self.turn["plan_mentions_language"] = plan_mentions_language
        # ROUNDTRIP
        self.turn["prior_plan"] = prior_plan
        self.turn["plan_plan"] = plan_plan
        self.turn["prior_partner_ref"] = prior_partner_ref
        self.turn["plan_partner_ref"] = plan_partner_ref
        # BEAM SEARCH OUTPUT
        self.turn["prior_beam_sents"] = prior_beam_sents
        self.turn["prior_beam_ref_res"] = prior_beam_ref_res
        self.turn["prior_beam_lm"] = prior_beam_lm
        self.turn["plan_beam_sents"] = plan_beam_sents
        self.turn["plan_beam_ref_res"] = plan_beam_ref_res
        self.turn["plan_beam_lm"] = plan_beam_lm
        self.turn["plan_beam_seed"] = plan_beam_seed
        # PLAN FEATURE LABEL
        self.turn["plan_prior"] = plan_prior
        self.turn["writer_configs_prior"] = writer_configs_prior
        self.turn["reader_configs_prior"] = reader_configs_prior
        self.turn["label_prior"] = label_prior

        self.turn["plan"] = plan
        self.turn["writer_configs"] = writer_configs
        self.turn["reader_configs"] = reader_configs
        self.turn["label"] = label

    def add_turn_resp(
        self,
        response_language,
        response,
        belief,
        configs,
        marginal_belief,
        response_utt=None,
        response_label=None,
        response_logits=None,
    ):
        self.turn["response_language"] = response_language
        self.turn["response"] = response
        self.turn["belief"] = belief.tolist()
        self.turn["configs"] = configs.tolist()
        self.turn["marginal_belief"] = marginal_belief.tolist()
        # SIMPLIFYING ASSUMP: ONLY A SINGLE UTT IN RESPONSE
        self.turn["response_utt"] = (
            response_utt.tolist()
            if response_utt is not None
            else None
        )
        self.turn["response_label"] = response_label
        self.turn["response_logits"] = response_logits


class StaticHierarchicalDialog(HierarchicalDialog):
    def __init__(
        self, agents, args, markable_detector,
        dialogues,
        markable_detector_corpus=None,
    ):
        super().__init__(
            agents, args, markable_detector,
            markable_detector_corpus=markable_detector_corpus,
        )
        # map from scenario to training conversation,
        # not sure if the logger relies on scenarios or something
        # TODO rewrite to use conversation id, since static
        self.dialogues = {
            x["scenario"]["uuid"]: x
            for x in dialogues
        }

    def run(self, ctxs, logger, max_words=5000):
        scenario_id = ctxs[0][0]
        print(f"Scenario id: {scenario_id}")
        self.dialog_logger = StaticDialogLogger(scenario_id, dir=self.args.dialog_log_dir)

        # setup for MBP
        dots = ctxs[2][0]
        their_dots = ctxs[2][1]
        num_dots = len(dots)
        state = [x in their_dots for x in dots]

        min_num_mentions = 0
        max_num_mentions = 10


        # lets just assume we go first
        YOU, THEM = 0,1

        # TODO(URGENT): DO WE NEED TO TOKENIZE?
        SENTENCES = [
            event["data"]
            for event in self.dialogues[scenario_id]["events"]
            if event["action"] == "message"
        ]

        #max_sentences = self.args.max_sentences
        max_sentences = len(SENTENCES)

        for agent in self.agents:
            assert [] == agent.model.args.ref_beliefs \
                   == agent.model.args.partner_ref_beliefs \
                   == agent.model.args.generation_beliefs \
                   == agent.model.args.selection_beliefs \
                   == agent.model.args.mention_beliefs
        belief_constructor = BlankBeliefConstructor()

        for agent, agent_id, ctx, real_ids in zip(self.agents, [0, 1], ctxs[1], ctxs[2]):
            agent.feed_context(ctx, belief_constructor)
            agent.real_ids = real_ids
            agent.agent_id = agent_id

            agent.dots = np.array(real_ids, dtype=int)
            # ctx: [x, y, size, color]

        device = self.agents[0].state.ctx_h.device

        # Agent 0 always goes first (static)
        # oh shit, this is wrong.
        # check first event in static dialogue!
        if self.dialogues[scenario_id]["events"][0]["agent"] == 0:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        is_selection_prediction = vars(writer.model.args).get('is_selection_prediction', False)
        is_selection_prediction_ = vars(reader.model.args).get('is_selection_prediction', False)
        if is_selection_prediction != is_selection_prediction_:
            raise NotImplementedError("both models must use --is_selection_prediction or not")

        conv = []
        speaker = []
        self.metrics.reset()

        words_left = max_words
        length = 0
        expired = False

        sentence_ix = 0

        while sentence_ix < max_sentences:
            assert writer.state.turn == sentence_ix
            assert reader.state.turn == sentence_ix

            if is_selection_prediction:
                is_selection_prob = writer.is_selection_outs[-1].sigmoid()
                this_is_selection = torch.distributions.Bernoulli(is_selection_prob).sample().bool().view((1,))
            else:
                this_is_selection = None

            this_partner_num_markables = torch.LongTensor([0])

            # WORDS FOR WRITING WITH FORCE_WORDS
            #out_words = "do you see the black one ? <eos>".split()
            #out_words = input("input utterance: ").split() + ["<eos>"]
            tokens = word_tokenize(SENTENCES[sentence_ix].lower().strip())
            out_words = tokens + ["<eos>"]
            #print(SENTENCES[sentence_ix])

            # PLAN ROUNDTRIP
            # prior language
            prior_lang = super(type(writer), writer).write(
                max_words=words_left,
                detect_markables=True,
                start_token='YOU:',
                is_selection=this_is_selection,
                inference=writer.args.language_inference,
                beam_size=writer.args.language_beam_size,
                sample_temperature_override=writer.args.language_sample_temperature,
                min_num_mentions=min_num_mentions,
                max_num_mentions=max_num_mentions,
                do_update = False,
            )

            # gather next mention predictions from writer
            nm_cands = writer.next_mention_candidates[-1]
            nm = writer.next_mention_predictions[-1]
            nm_multi = writer.next_mention_predictions_multi[-1]
            # int is easer to read
            nms = torch.stack([
                x.any(0)[0] for x in nm_multi
            ]).int() if nm_multi is not None else None

            # get reference predictions from reader
            super(type(reader), reader).read(
                prior_lang,
                dots_mentioned=None,
                dots_mentioned_per_ref=None,
                dots_mentioned_num_markables=this_partner_num_markables,
                detect_markables=True,
                is_selection=this_is_selection,
                do_update = False,
            )
            # THESE ARE THE DEVIATIONS
            # roundtrip words -> my refs
            prior_ref = (writer.dots_mentioned_per_ref_chosen[-1].int().cpu().numpy()
                if writer.dots_mentioned_per_ref_chosen[-1] is not None
                else None
            )
            # roundtrip words -> their refs
            prior_partner_ref = (reader.partner_ref_preds[-1].cpu().numpy()
                if reader.partner_ref_preds[-1] is not None
                else None
            )
            extra = writer.extras[-1]
            prior_beam_sents = [" ".join(xs) for xs in extra["words"]]
            prior_beam_ref_res = (extra["ref_resolution_scores"].tolist()
                if "ref_resolution_scores" in extra else None)
            prior_beam_lm = (extra["language_model_scores"].tolist()
                if "language_model_scores" in extra else None)

            # POP STATE, planning gets fresh state (from static dialog)
            undo_state_writer(writer)
            undo_state_reader(reader)
            # / PLAN ROUNDTRIP

            """
            # DBG: check if repeat is exactly the same
            # prior language
            prior_lang2 = super(type(writer), writer).write(
                max_words=words_left,
                detect_markables=True,
                start_token='YOU:',
                is_selection=this_is_selection,
                inference=writer.args.language_inference,
                beam_size=writer.args.language_beam_size,
                sample_temperature_override=writer.args.language_sample_temperature,
                min_num_mentions=min_num_mentions,
                max_num_mentions=max_num_mentions,
                do_update = False,
            )

            # gather next mention predictions from writer
            nm_cands2 = writer.next_mention_candidates[-1]
            nm2 = writer.next_mention_predictions[-1]
            nm_multi2 = writer.next_mention_predictions_multi[-1]
            # int is easer to read
            nms2 = torch.stack([
                x.any(0)[0] for x in nm_multi
            ]).int() if nm_multi is not None else None

            # POP STATE, planning gets fresh state (from static dialog)
            undo_state_writer(writer)
            import pdb; pdb.set_trace()
            # / DBG
            """

            # next mention predictions from planning
            if isinstance(writer, BeliefAgent):
                EdHs = writer.belief.compute_EdHs(writer.prior)
                cs, hs = writer.belief.viz_belief(EdHs, n=writer.args.next_mention_reranking_k)
            else:
                cs = None

            # plan feature level labels
            # get the plan resolution sets for prior model
            plan_prior = prior_ref[0].any(0).astype(int)
            feats_prior = writer.belief.get_feats(plan_prior)
            writer_matches_prior = writer.belief.resolve_utt(*feats_prior)
            reader_matches_prior = reader.belief.resolve_utt(*feats_prior)
            writer_configs_prior = writer.dots[writer_matches_prior]
            reader_configs_prior = reader.dots[reader_matches_prior]
            label_prior = label_config_sets(writer_configs_prior, reader_configs_prior)

            # get the plan resolution sets for planning model
            plan = cs[0]
            feats = writer.belief.get_feats(plan)
            writer_matches = writer.belief.resolve_utt(*feats)
            reader_matches = reader.belief.resolve_utt(*feats)
            writer_configs = writer.dots[writer_matches]
            reader_configs = reader.dots[reader_matches]
            label = label_config_sets(writer_configs, reader_configs)
            # / plan feature level labels

            # PLAN ROUNDTRIP
            plan_lang = writer.write_supervised(
                max_words=words_left,
                detect_markables=True,
                start_token='YOU:',
                is_selection=this_is_selection,
                inference=writer.args.language_inference,
                beam_size=writer.args.language_beam_size,
                sample_temperature_override=writer.args.language_sample_temperature,
                min_num_mentions=min_num_mentions,
                max_num_mentions=max_num_mentions,
                do_update = False,
            )
            # get reference predictions from reader
            super(type(reader), reader).read(
                plan_lang,
                dots_mentioned=None,
                dots_mentioned_per_ref=None,
                dots_mentioned_num_markables=this_partner_num_markables,
                detect_markables=True,
                is_selection=this_is_selection,
                do_update = False,
            )
            # THESE ARE THE DEVIATIONS
            # roundtrip words -> my refs
            plan_ref = (writer.dots_mentioned_per_ref_chosen[-1].int().cpu().numpy()
                if writer.dots_mentioned_per_ref_chosen[-1] is not None
                else None
            )
            # roundtrip words -> their refs
            plan_partner_ref = (reader.partner_ref_preds[-1].cpu().numpy()
                if reader.partner_ref_preds[-1] is not None
                else None
            )
            extra = writer.extras[-1]
            plan_beam_sents = [" ".join(xs) for xs in extra["words"]]
            plan_beam_ref_res = (extra["ref_resolution_scores"].tolist()
                if "ref_resolution_scores" in extra else None)
            plan_beam_lm = (extra["language_model_scores"].tolist()
                if "language_model_scores" in extra else None)
            plan_beam_seed = writer.next_mention_plans[-1].tolist()

            # POP OFF STATE TO RESUME STATIC DIALOG
            undo_state_writer(writer)
            undo_state_reader(reader)
            # / PLAN ROUNDTRIP

            # WRITER, force words
            writer.args.reranking_confidence = False
            writer.write_supervised(
                force_words=[out_words],
                start_token="YOU:",
                detect_markables=True,
                is_selection=this_is_selection,
                inference="sample",
            )
            writer.args.reranking_confidence = True
            # check writer.ref_outs and writer.ref_preds
            #import pdb; pdb.set_trace()

            # LOGGING
            #if writer.agent_id == YOU:
            if True:
                print(f"{writer.name} writer dots")
                print(writer.dots)
                print("prior next mentions")
                print(nms)
                print("mbp next mentions")
                print(cs)

                print("prior language")
                print(" ".join(prior_lang))
                print("prior plan")
                print(prior_ref)
                print("prior partner ref")
                print(prior_partner_ref)
                print("plan language")
                print(" ".join(plan_lang))
                print("plan plan")
                print(plan_ref)
                print("plan partner ref")
                print(plan_partner_ref)

                #self.dialog_logger.start_turn(YOU)
                self.dialog_logger.start_turn(
                    writer.agent_id,
                    reader.agent_id,
                )
                self.dialog_logger.add_turn_utt(
                    writer.belief.configs.tolist(),
                    utterance_language = SENTENCES[sentence_ix],
                    utterance = writer.ref_preds[-1].any(0)[0].int().tolist()
                        if writer.ref_preds[-1] is not None
                        else None,
                    prior_mentions = nms.tolist() if nms is not None else None,
                    plan_mentions = cs.tolist(),
                    prior_mentions_language = " ".join(prior_lang),
                    plan_mentions_language = " ".join(plan_lang),
                    prior_plan = prior_ref.tolist(),
                    plan_plan = plan_ref.tolist(),
                    prior_partner_ref = prior_partner_ref.tolist()
                        if prior_partner_ref is not None else None,
                    plan_partner_ref = plan_partner_ref.tolist()
                        if plan_partner_ref is not None else None,
                    # BEAM SEARCH
                    prior_beam_sents = prior_beam_sents,
                    prior_beam_ref_res = prior_beam_ref_res,
                    prior_beam_lm = prior_beam_lm,
                    plan_beam_sents = plan_beam_sents,
                    plan_beam_ref_res = plan_beam_ref_res,
                    plan_beam_lm = plan_beam_lm,
                    plan_beam_seed = plan_beam_seed,
                    # PLAN FEATURE EVALUATION
                    plan_prior = plan_prior.tolist(),
                    writer_configs_prior = writer_configs_prior.tolist(),
                    reader_configs_prior = reader_configs_prior.tolist(),
                    label_prior = label_prior.value,
                    plan = plan.tolist(),
                    writer_configs = writer_configs.tolist(),
                    reader_configs = reader_configs.tolist(),
                    label = label.value,
                )

            # READER
            reader.read(
                out_words,
                dots_mentioned=None,
                dots_mentioned_per_ref=None,
                dots_mentioned_num_markables=this_partner_num_markables,
                detect_markables=True,
                is_selection=this_is_selection,
            )
            # check reader.partner_ref_outs and writer.partner_ref_preds
            #import pdb; pdb.set_trace

            # MBP
            #if reader.agent_id == YOU:
            if True:
                # UPDATE AGENT 0 BELIEF
                print(f"READER NAME {reader.name}")
                response = None
                if sentence_ix > 0 and reader.ref_preds[-2] is not None:
                    # update belief given partner response to our utterances
                    #print(reader.ref_preds[-1].any(0)[0].int().tolist())
                    #print(reader.ref_preds[-2].any(0)[0].int().tolist())
                    utt = reader.ref_preds[-2].any(0)[0].int().cpu().numpy()
                    label = reader.responses[-1]
                    #if label != 0:
                        #response = 1 if label == 1 else 0
                        #reader.prior1 = reader.belief1.posterior(reader.prior1, utt, response)
                        #reader.prior2 = reader.belief2.posterior(reader.prior2, utt, response)
                        #reader.prior3 = reader.belief3.posterior(reader.prior3, utt, response)
                    utt_str = np.array(reader.dots)[utt.astype(bool)]
                    print("our prev utt")
                    print(utt_str)
                    print("response")
                    print(label)
                """
                else:
                    # if the first turn is a read, give null data for utts
                    self.dialog_logger.start_turn(YOU)
                    self.dialog_logger.add_turn_utt(
                        writer.belief.configs.tolist(),
                        utterance_language = None,
                        utterance = None,
                        prior_mentions = None,
                        plan_mentions = None,
                        plan2_mentions = None,
                        plan3_mentions = None,
                        prior_mentions_language = None,
                        plan_mentions_language = None,
                        plan2_mentions_language = None,
                        plan3_mentions_language = None,
                    )
                """

                #side_info = reader.side_infos[-1]
                #if side_info is not None and side_info.any():
                #if UTTS[sentence_ix] is not None:
                    # update belief with unsolicited partner info
                    # if they mention a new dot, we pretend we asked about it
                    # reader.prior is updated in reader.read()
                    #reader.prior1 = reader.belief1.posterior(reader.prior1, side_info, 1)
                    #reader.prior2 = reader.belief2.posterior(reader.prior2, side_info, 1)
                    #reader.prior3 = reader.belief3.posterior(reader.prior3, side_info, 1)

                if isinstance(reader, BeliefAgent):
                    cs, ps = reader.belief.viz_belief(reader.prior, n=5)
                    print("posterior")
                    print(cs)
                    print(ps)
                    print("marginals")
                    marginals = reader.belief.marginals(reader.prior)
                    print([f"{x:.2f}" for x in marginals])
                else:
                    ps, cs, marginals = None, None, None

                their_utt = None
                if reader.partner_ref_preds[-1] is not None:
                    their_utt = reader.partner_ref_preds[-1].any(0)[0].cpu().numpy()
                    utt_str = np.array(reader.dots)[their_utt]
                    print("their utt")
                    print(utt_str)

                self.dialog_logger.add_turn_resp(
                    #response_language = SENTENCES[sentence_ix+1]
                        #if sentence_ix < len(SENTENCES)-1 else None,
                    response_language = None,
                    response = response,
                    belief = ps,
                    configs = cs,
                    marginal_belief = marginals,
                    response_utt = their_utt.astype(int)
                        if their_utt is not None
                        else None,
                    response_label = reader.responses[-1],
                    response_logits = reader.response_logits[-1],
                )
                self.dialog_logger.end_turn()
            # / MBP

            words_left -= len(out_words)
            length += len(out_words)

            self.metrics.record('sent_len', len(out_words))
            if 'full_match' in self.metrics.metrics:
                self.metrics.record('full_match', out_words)
            self.metrics.record('%s_unique' % writer.name, out_words)

            conv.append(out_words)
            speaker.append(writer.agent_id)

            # if not writer.human:
            if True:
                logger.dump_sent(writer.name, out_words)
                # if writer.decoded_beam_best:
                #     logger.dump_sent(writer.name + ' beam best', writer.decoded_beam_best[-1])

            if logger.scenarios and self.args.log_attention:
                attention = writer.get_attention()
                if attention is not None:
                    logger.dump_attention(writer.name, writer.agent_id, scenario_id, attention)

            if self._is_selection(out_words):
                self.metrics.record('%s_make_sel' % writer.name, 1)
                self.metrics.record('%s_make_sel' % reader.name, 0)
                break

            if words_left <= 1:
                break

            writer, reader = reader, writer
            sentence_ix += 1

        self.dialog_logger.dump_json()

        choices = []
        for agent in self.agents:
            choice = agent.choose()
            choices.append(choice)
        if logger.scenarios:
            logger.dump_choice(scenario_id, choices)
            if "plot_metrics" in self.args and self.args.plot_metrics:
                for agent in [0, 1]:
                    for obj in logger.scenarios[scenario_id]['kbs'][agent]:
                        if obj['id'] == choices[agent]:
                            self.metrics.record('%s_sel_bias' % writer.name, obj,
                                                logger.scenarios[scenario_id]['kbs'][agent])

        agree, rewards = self.domain.score_choices(choices, ctxs)
        if expired:
            agree = False
        logger.dump('-' * 80)
        logger.dump_agreement(agree)
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            j = 1 if i == 0 else 0
            agent.update(agree, reward, choice=choices[i])

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
            self.metrics.record('moving_advantage', rewards[0] - rewards[1])
            self.metrics.record('agree_comb_rew', np.sum(rewards))
            for agent, reward in zip(self.agents, rewards):
                self.metrics.record('agree_%s_rew' % agent.name, reward)

        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('moving_agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)
            self.metrics.record('%s_moving_rew' % agent.name, reward if agree else 0)

        for agent in self.agents:
            if agent.ref_resolution_scores:
                self.metrics.record('%s_ref_resolution_score' % agent.name, np.mean(agent.ref_resolution_scores))
            if agent.language_model_scores:
                self.metrics.record('%s_language_model_score' % agent.name, np.mean(agent.language_model_scores))
            if agent.joint_scores:
                self.metrics.record('%s_joint_score' % agent.name, np.mean(agent.joint_scores))

        if self.markable_detector is not None and self.markable_detector_corpus is not None:
            markable_list = []
            referents_dict = {}

            markable_starts = []
            for agent in [0, 1]:
                dialog_tokens = []
                dialog_text = ""
                for spkr, uttr in zip(speaker, conv):
                    if spkr == agent:
                        dialog_tokens.append("YOU:")
                    else:
                        dialog_tokens.append("THEM:")
                    dialog_tokens += uttr
                    dialog_text += str(spkr) + ": " + " ".join(uttr[:-1]) + "\n"

                    referent_inputs = []
                    markables = []
                    markable_ids = []
                    # TODO: fix this; detect_markables should probably only be run once per dialogue
                    for markable, referent_inp in self.markable_detector.detect_markables(
                        dialog_tokens,
                        dialog_text=dialog_text,
                        device=torch.device('cuda') if self.args.cuda else None,
                    ):
                        if not markable['is_self']:
                            continue
                        markable['speaker'] = agent
                        markable_id = len(markable_list)
                        markable['markable_id'] = markable_id
                        markable_list.append(markable)
                        markable_ids.append(markable_id)
                        referent_inputs.append(referent_inp)

                    ref_out = self.agents[agent].predict_referents(referent_inputs)

                    if ref_out is not None:
                        for i, markable_id in enumerate(markable_ids):
                            ent_ids = [ent["id"] for ent in logger.scenarios[scenario_id]['kbs'][agent]]
                            referents = []
                            for j, is_referent in enumerate((ref_out[i] > 0).tolist()):
                                if is_referent:
                                    referents.append("agent_" + str(agent) + "_" + ent_ids[j])

                            referents_dict[markable_id] = referents

            #markable_starts = list(set(markable_starts))
            # reindex markable ids
            markable_id_and_start = [(markable_id, markable_start) for markable_id, markable_start in zip(range(len(markable_starts)), markable_starts)]
            reindexed_markable_ids = [markable_id for markable_id, _ in sorted(markable_id_and_start, key = lambda x: x[1])]

            self.selfplay_markables[scenario_id] = {}
            self.selfplay_referents[scenario_id] = {}

            # add markables
            self.selfplay_markables[scenario_id]["markables"] = []
            for new_markable_id, old_markable_id in enumerate(reindexed_markable_ids):
                markable = markable_list[old_markable_id]
                markable["markable_id"] = "M{}".format(new_markable_id + 1)
                self.selfplay_markables[scenario_id]["markables"].append(markable)

            # add dialogue text
            self.selfplay_markables[scenario_id]["text"] = dialog_text

            # add final selections
            self.selfplay_markables[scenario_id]["selections"] = choices

            # add referents
            for new_markable_id, old_markable_id in enumerate(reindexed_markable_ids):
                referents = referents_dict[old_markable_id]
                self.selfplay_referents[scenario_id]["M{}".format(new_markable_id + 1)] = referents

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        #for ctx, choice in zip(ctxs, choices):
        #    logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards


