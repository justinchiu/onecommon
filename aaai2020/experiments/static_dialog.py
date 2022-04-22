import sys

import numpy as np

import torch

from engines.beliefs import BlankBeliefConstructor
from metric import MetricsContainer
from corpora import data
import domain
from dialog import DialogLogger, HierarchicalDialog

from belief import AndOrBelief, OrAndBelief

class StaticHierarchicalDialog(HierarchicalDialog):
    def __init__(
        self, agents, args, markable_detector,
        markable_detector_corpus=None,
    ):
        super().__init__(
            agents, args, markable_detector,
            markable_detector_corpus=markable_detector_corpus,
        )

    def run(self, ctxs, logger, max_words=5000):
        scenario_id = ctxs[0][0]

        # setup for MBP
        dots = ctxs[2][0]
        their_dots = ctxs[2][1]
        num_dots = len(dots)
        state = [x in their_dots for x in dots]

        min_num_mentions = 0
        max_num_mentions = 10


        SENTENCES = [
            "do you see two black dots close together ?",
            "yes .",
            "is one below and to the right of the other ?",
            "no .",
            "do you see a triangle of three grey dots ?",
            "yes .",
        ]
        UTTS = [
            np.array([0,0,1,0,0,1,0]),
            np.array([0,0,1,0,0,1,0]),
            np.array([0,0,0,1,1,0,1]),
        ]
        RESPS = [
            1,
            0,
            1,
        ]
        # / MBP

        max_sentences = self.args.max_sentences

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

            agent.belief = OrAndBelief(num_dots)
            agent.prior = agent.belief.prior
            agent.dots = ctxs[2][agent_id]

        device = self.agents[0].state.ctx_h.device

        # Agent 0 always goes first (static)
        writer, reader = self.agents

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
        read_idx = 0

        while sentence_ix < max_sentences:
            assert writer.state.turn == sentence_ix
            assert reader.state.turn == sentence_ix

            if is_selection_prediction:
                is_selection_prob = writer.is_selection_outs[-1].sigmoid()
                this_is_selection = torch.distributions.Bernoulli(is_selection_prob).sample().bool().view((1,))
            else:
                this_is_selection = None

            # BEGIN COPY

            this_partner_num_markables = torch.LongTensor([0])


            WRITE = False
            if WRITE:
                out_words = writer.write(
                    max_words=words_left,
                    detect_markables=True,
                    start_token='YOU:',
                    is_selection=this_is_selection,
                    inference=self.args.language_inference,
                    beam_size=self.args.language_beam_size,
                    sample_temperature_override=self.args.language_sample_temperature,
                    min_num_mentions=min_num_mentions,
                    max_num_mentions=max_num_mentions,
                )
                # pop off reader_lang_hs, writer_lang_hs
                writer.reader_lang_hs.pop()
                writer.writer_lang_hs.pop()
            else:            
                # gather next mention predictions from writer
                if writer.reader_lang_hs:
                    is_self = torch.ones(1).bool()
                    can_confirm = writer.make_can_confirm(
                        is_self,
                        vars(writer.model.args).get("confirmations_resolution_strategy", "any"),
                    )
                    writer.next_mention(
                        lens=torch.LongTensor([writer.reader_lang_hs[-1].size(0)]).to(writer.device),
                        dots_mentioned_num_markables_to_force=None,
                        min_num_mentions=0,
                        max_num_mentions=12,
                        can_confirm=can_confirm,
                    )
                else:
                    writer.first_mention(
                        dots_mentioned_num_markables_to_force=None,
                        min_num_mentions=0,
                        max_num_mentions=12,
                    )
            nm = writer.next_mention_predictions
            nm_multi = writer.next_mention_predictions_multi
            # int is easer to read
            nms = torch.stack([x.any(0)[0] for x in nm_multi[0]]).int()

            # next mention predictions from planning
            EdHs = writer.belief.compute_EdHs(writer.prior)
            cs, hs = writer.belief.viz_belief(EdHs, n=writer.args.next_mention_reranking_k)

            if writer.agent_id == 0:
                print("writer dots")
                print(writer.dots)
                print("prior next mentions")
                print(nms)
                print("mbp next mentions")
                print(cs)
                #import pdb; pdb.set_trace()


            # update state for next turn
            #out_words = "do you see the black one ? <eos>".split()
            #out_words = input("input utterance: ").split() + ["<eos>"]
            out_words = SENTENCES[sentence_ix].split() + ["<eos>"]
            print(SENTENCES[sentence_ix])

            if reader.agent_id == 0:
                # UPDATE AGENT 0 BELIEF
                utt = UTTS[read_idx]
                response = RESPS[read_idx]
                reader.prior = reader.belief.posterior(reader.prior, utt, response)
                read_idx += 1

                utt_str = np.array(reader.dots)[utt.astype(bool)]
                cs, ps = reader.belief.viz_belief(reader.prior, n=5)
                print("prev utt")
                print(utt_str)
                print("posterior")
                print(cs)
                print(ps)
                import pdb; pdb.set_trace()

            # WRITER
            writer.read(
                out_words,
                start_token="YOU:",
                dots_mentioned=None,
                dots_mentioned_per_ref=None,
                dots_mentioned_num_markables=this_partner_num_markables,
                detect_markables=True,
                is_selection=this_is_selection,
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
