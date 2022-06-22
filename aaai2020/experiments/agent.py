import sys
from collections import defaultdict
import pdb

import numpy as np
import torch
from torch import optim, autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from corpora.reference import PRONOUNS
from dialog import DialogLogger
import domain
from engines import Criterion
import math
from collections import Counter, namedtuple

from nltk.parse import CoreNLPParser, CoreNLPDependencyParser

from models import RnnReferenceModel, HierarchicalRnnReferenceModel
from models.markable_detector import BiLSTM_CRF
import models.reference_predictor
from models.reference_predictor import RerankingMentionPredictor, ReferencePredictor
from models.rnn_reference_model import State, NextMentionLatents
from engines.rnn_reference_engine import make_can_confirm_single
from models.ctx_encoder import pairwise_differences

# POLICY BLOCK
#import pomdp_py
#from pomdp_py.utils import TreeDebugger
#from mab.beta_bernoulli.domain.action import Ask, Select
#from mab.beta_bernoulli.domain.observation import ProductObservation
#from mab.beta_bernoulli.dialogue import take_turn, plan
#from mab.beta_bernoulli.problem import RankingAndSelectionProblem, belief_update
# / POLICY BLOCK

GenerationOutput = namedtuple("GenerationOutput", [
    "dots_mentioned_per_ref",
    "num_markables",
    "outs",
    "logprobs",
    "state",
    "reader_and_writer_lang_hs",
    "extra",
])

class Agent(object):
    """ Agent's interface. """
    def feed_context(self, ctx):
        pass
    def read(self, inpt):
        pass

    def write(self):
        pass

    def choose(self):
        pass

    def update(self, agree, reward, choice):
        pass

    def get_attention(self):
        return None


YOU_TOKEN = 'YOU:'
THEM_TOKEN = 'THEM:'

class RnnAgent(Agent):

    @staticmethod
    def add_args(parser):
        # default to pragmatic confidence args https://github.com/justinchiu/onecommon/blob/master/webapp/src/web/app_params.json.full#L53
        parser.add_argument('--language_inference', choices=['beam', 'noised_beam', 'sample', 'forgetful_noised_beam'], default='forgetful_noised_beam')
        parser.add_argument('--language_beam_size', type=int, default=100)
        parser.add_argument('--language_beam_keep_all_finished', action='store_true')
        # do the above ^
        parser.add_argument('--language_sample_temperature', type=float, default=0.25)
        parser.add_argument('--language_rerank', action='store_true')
        parser.add_argument('--language_rerank_weight', type=float, default=.999)
        parser.add_argument('--max_sentences', type=int, default=20)

        parser.add_argument('--next_mention_reranking', action='store_true')
        parser.add_argument('--next_mention_reranking_max_mentions', type=int, default=10)
        parser.add_argument('--next_mention_reranking_k', default=20, type=int)
        parser.add_argument('--next_mention_reranking_score',
                            default='log_max_probability',
                            choices=RerankingMentionPredictor.SCORING_FUNCTIONS)
        parser.add_argument('--next_mention_reranking_weight', default=0, type=float)
        parser.add_argument('--next_mention_candidate_generation',
                            choices=['topk', 'topk_multi_mention', 'sample'], default='topk')
        parser.add_argument('--next_mention_reranking_use_stop_scores', action='store_true')

        parser.add_argument('--reranking_confidence', action='store_true')
        parser.add_argument('--reranking_confidence_exp_score_threshold', type=float, default=0.8)
        parser.add_argument('--reranking_confidence_score', choices=['l0', 'joint'], default='joint')
        parser.add_argument('--reranking_confidence_type',
                            choices=['first_above_threshold', 'keep_best'],
                            default='first_above_threshold')
        # POLICY ARGS
        #parser.add_argument(
            #'--policy',
            #choices=['rnn', 'beta_bernoulli'],
            #default='rnn',
        #)
        # MBP
        parser.add_argument('--DBG_GEN', action='store_true')
        parser.add_argument('--DBG_PLAN', default="", type=str,
            help="Output file path for top-k dot configurations during each turn at planning",
        )
        # / MBP

    def __init__(
        self, model: HierarchicalRnnReferenceModel,
        args, name='Alice', train=False, markable_detector=None,
    ):
        super(RnnAgent, self).__init__()
        self.model: HierarchicalRnnReferenceModel = model
        self.reference_predictor = ReferencePredictor(model.args)
        self.markable_detector: BiLSTM_CRF = markable_detector
        self.args = args
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)
        self.device = 'cuda' if args.cuda else 'cpu'
        self.train = train
        self.selection_word_index = self.model.word_dict.get_idx('<selection>')
        if self.args.next_mention_reranking:
            self.next_mention_predictor = RerankingMentionPredictor(
                args, scoring_function=self.args.next_mention_reranking_score,
                default_weight=self.args.next_mention_reranking_weight,
                weights=[self.args.next_mention_reranking_weight],
            )
        else:
            self.next_mention_reranking_k = None
            self.next_mention_predictor = ReferencePredictor(args)
        if train:
            raise NotImplementedError("fix optimization")
            self.model.train()
            self.opt = optim.RMSprop(
                self.model.parameters(),
                lr=args.rl_lr,
                momentum=self.args.momentum
            )
            self.all_rewards = []
            self.t = 0
        else:
            self.model.eval()
        # print("language_rerank: {}".format(self.args.language_rerank))
        #
        #
        # POLICY BLOCK
        #self.policy = args.policy
        #if self.name == "Alice":
        #if True:
        #if False:
            #self.policy = "beta_bernoulli" # JUST FOR DEBUGGING

            # random fake ground truth vector here
            #dots = np.zeros(7, dtype=np.bool)
            #dots[:4] = True
            # problem is mostly a datastructure for holding the agent
            #self.problem = RankingAndSelectionProblem(
                #dot_vector = dots,
                #max_turns = 20,
                #belief_rep = "particles",
                #num_bins=2,
                #num_particles = 0,
                #enumerate_belief = True,
            #)
            #self.planner = planner = pomdp_py.POMCP(
                #max_depth = 20, # need to change
                #discount_factor = 1,
                #num_sims = 10000,
                #exploration_const = 100,
                #rollout_policy = problems[0].agent.policy_model, # need to change per agent?
                #num_rollouts=1,
            #)
        # / POLICY BLOCK

    def _encode(self, inpt, dictionary):
        encoded = torch.Tensor(dictionary.w2i(inpt)).long().unsqueeze(1)
        return encoded

    def _decode(self, out, dictionary):
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context, belief_constructor=None):
        self.reader_lang_hs = []
        self.writer_lang_hs = []
        self.logprobs = []
        self.sents = []
        # self.sents_beam_best = []
        self.words = []
        # self.words_beam_best = []
        self.decoded_beam_best = []
        self.context = context
        ctx = torch.Tensor([float(x) for x in context]).float().unsqueeze(0) # add batch size of 1
        self.state: State = self.model.initialize_state(ctx, belief_constructor)
        self.timesteps = 0

        # for use with predict_referents
        self.ref_inpts = []
        self.markables = []
        self.partner_ref_inpts = []
        self.partner_markables = []

        # for use with the belief constructor
        self.ref_outs = []
        self.ref_preds = []
        self.partner_ref_outs = []
        self.partner_ref_preds = []
        self.next_mention_latents = []
        self.next_mention_candidates = []
        self.next_mention_predictions = []
        self.next_mention_predictions_multi = []
        self.dots_mentioned_per_ref_chosen = []
        self.next_mention_indices_sorted = []

        self.joint_scores = []
        self.language_model_scores = []
        self.ref_resolution_scores = []

        self.is_selection_outs = [self.model.is_selection_prediction(self.state)]
        self.sel_outs = []
        self.extras = []

        # POLICY BLOCK
        #if self.policy == "beta_bernoulli":
            # reset tree and history
            #self.actions = []
            #self.problem.agent.reset_tree_and_history()
        # / POLICY BLOCK

    def feed_partner_context(self, partner_context):
        pass

    def next_mention_prediction_and_candidates_from_latents(self, state: State, next_mention_latents: NextMentionLatents):
        # JC: RNN NEXT MENTION COMES FROM HERE
        dummy_gold_mentions = torch.zeros(
            (state.bsz, next_mention_latents.dots_mentioned_num_markables.max().item(), self.model.num_ent)
        ).long().to(self.device)
        if self.args.next_mention_reranking:
            # hack; pass True for inpt because this method only uses it to ensure it's not null
            # next_mention_scores, stop_loss, nm_num_markables = self.model.next_mention_prediction_from_latents(
            #     state, next_mention_latents
            # )
            candidates = self.model.rollout_next_mention_cands(
                state, next_mention_latents, num_candidates=self.args.next_mention_reranking_k,
                generation_method=self.args.next_mention_candidate_generation,
                use_stop_losses=self.args.next_mention_reranking_use_stop_scores,
            )
            # num_markables_per_candidate = nm_num_markables.unsqueeze(1).expand(-1, self.args.next_mention_reranking_k)
            _loss, predictions, nm_num_markables, _stats, predictions_multi_sorted, nm_num_markables_multi_sorted, indices_sorted = self.next_mention_predictor.forward(
                True, dummy_gold_mentions, None, candidates.num_markables_per_candidate,
                candidates,
                # TODO: collapse param naming is misleading; collapse adds in additional expanded_* terms
                collapse=self.model.args.next_mention_prediction_type=='multi_reference',
            )
        else:
            next_mention_scores, stop_loss, nm_num_markables = self.model.next_mention_prediction_from_latents(
                state, next_mention_latents
            )
            if next_mention_scores is None:
                return None, None, None, None
            # hack; pass True for inpt because this method only uses it to ensure it's not null
            _loss, predictions, _stats = self.next_mention_predictor.forward(
                True, dummy_gold_mentions, next_mention_scores, nm_num_markables,
                collapse=self.model.args.next_mention_prediction_type=='multi_reference',
            )
            predictions_multi_sorted = None
            nm_num_markables_multi_sorted = None
            indices_sorted = None
            candidates = None
        predictions_truncated = predictions[:nm_num_markables.item()] if predictions is not None else predictions
        predictions_multi_sorted_truncated = [
            pred[:nm.item()] for pred, nm in zip(predictions_multi_sorted, nm_num_markables_multi_sorted)
        ] if predictions_multi_sorted is not None else None
        return predictions_truncated, candidates, predictions_multi_sorted_truncated, indices_sorted

    def predict_referents(self, ref_inpt, num_markables):
        # JC: what is this doing? seems like it operates on next mention candidates.
        ref_beliefs = self.state.make_beliefs('ref', self.timesteps, self.partner_ref_outs, self.ref_outs)
        ref_out = self.model.reference_resolution(
            self.state, self.reader_lang_hs[-1], ref_inpt, num_markables,
            for_self=True, ref_beliefs=ref_beliefs
        )
        self.ref_outs.append(ref_out)

        dummy_targets = torch.zeros((1, num_markables.item(), 7)).long()
        _, ref_preds, ref_stats = self.reference_predictor.forward(
            ref_inpt, dummy_targets, ref_out, num_markables
        )
        # TEMPLATE
        self.ref_preds.append(ref_preds)
        return ref_out

    def predict_partner_referents(self, partner_ref_inpt, partner_num_markables):
        partner_ref_beliefs = self.state.make_beliefs('partner_ref', self.timesteps, self.partner_ref_outs, self.ref_outs)
        partner_ref_out = self.model.reference_resolution(
            self.state, self.reader_lang_hs[-1], partner_ref_inpt, partner_num_markables,
            for_self=False, ref_beliefs=partner_ref_beliefs
        )
        self.partner_ref_outs.append(partner_ref_out)

        dummy_targets = torch.zeros((1, partner_num_markables.item(), 7)).long()
        _, ref_preds, ref_stats = self.reference_predictor.forward(
            partner_ref_inpt, dummy_targets, partner_ref_out, partner_num_markables
        )
        self.partner_ref_preds.append(ref_preds)
        # JC: this is where the reference resolution is when reading!
        return partner_ref_out

    def update_dot_h(self, ref_inpt, partner_ref_inpt, num_markables, partner_num_markables,
                     ref_tgt=None, partner_ref_tgt=None):
        self.state = self.model._update_dot_h_maybe_multi(
            self.state, self.reader_lang_hs[-1],
            ref_inpt, partner_ref_inpt,
            num_markables, partner_num_markables,
            self.ref_outs[-1], self.partner_ref_outs[-1],
            ref_tgt, partner_ref_tgt,
        )

    def detect_markables(self, utterance_words, remove_pronouns=False):
        """-> markables: List, ref_boundaries: List"""
        if utterance_words[0] == THEM_TOKEN:
            utterance_words = [YOU_TOKEN] + utterance_words[1:]
        elif utterance_words[0] == YOU_TOKEN:
            pass
        else:
            utterance_words = [YOU_TOKEN] + utterance_words
        markables = []
        ref_boundaries = []
        for markable, ref_boundary in self.markable_detector.detect_markables(utterance_words):
            if remove_pronouns:
                if PRONOUNS & set(markable['text'].split()):
                    continue
            markables.append(markable)
            ref_boundaries.append(ref_boundary)
        return markables, ref_boundaries

    def markables_to_tensor(self, ref_boundaries):
        partner_num_markables = torch.LongTensor([len(ref_boundaries)]).to(self.device)
        if len(ref_boundaries) > 0:
            # add batch dimension
            return torch.LongTensor(ref_boundaries).unsqueeze(0).to(self.device), partner_num_markables
        else:
            return None, partner_num_markables

    def read(self, inpt_words, dots_mentioned=None, dots_mentioned_per_ref=None,
             dots_mentioned_num_markables=None,
             start_token=THEM_TOKEN,
             partner_ref_inpt=None, partner_num_markables=None,
             ref_tgt=None, partner_ref_tgt=None,
             detect_markables=False,
             is_selection=None,
             can_confirm=None, do_update = True,
             ):

        if can_confirm is None:
            is_self = torch.zeros(1).bool()
            can_confirm = self.make_can_confirm(is_self, vars(self.model.args).get("confirmations_resolution_strategy", "any"))

        self.sents.append(Variable(self._encode([start_token] + inpt_words, self.model.word_dict)))
        inpt = self._encode(inpt_words, self.model.word_dict)
        if self.model.args.feed_context_attend:
            raise NotImplementedError("need to detect markables and pass those as dots_mentioned (if this was reading YOU:; currently not)")

        generation_beliefs = self.state.make_beliefs('generation', self.timesteps, self.partner_ref_outs, self.ref_outs)
        (reader_lang_hs, writer_lang_hs), self.state = self.model.read(
            self.state, Variable(inpt),
            prefix_token=start_token,
            dots_mentioned=dots_mentioned,
            dots_mentioned_per_ref=dots_mentioned_per_ref,
            dots_mentioned_num_markables=dots_mentioned_num_markables,
            is_selection=is_selection,
            generation_beliefs=generation_beliefs,
            can_confirm=can_confirm,
        )
        self.reader_lang_hs.append(reader_lang_hs)
        self.writer_lang_hs.append(writer_lang_hs)

        if detect_markables:
            assert (not self.model.args.dot_recurrence_oracle) or ('partner' not in self.model.args.dot_recurrence_oracle_for)
            assert self.markable_detector is not None
            partner_markables, ref_boundaries = self.detect_markables(
                self._decode(inpt, self.model.word_dict), remove_pronouns=False
            )
            partner_ref_inpt, partner_num_markables = self.markables_to_tensor(ref_boundaries)
            partner_ref_tgt = torch.zeros((1, partner_num_markables.item(), 7)).long()

            self.ref_inpts.append(None)
            self.markables.append([])
            self.partner_ref_inpts.append(partner_ref_inpt)
            self.partner_markables.append(partner_markables)
            # JC: doesnt look like mention input is here?

        if self.model.args.partner_reference_prediction and partner_ref_inpt is not None and partner_num_markables is not None:
            self.predict_partner_referents(partner_ref_inpt, partner_num_markables)

            # POLICY BLOCK
            #if self.policy == "beta_bernoulli":
                #agent = self.problem.agent
                # JC: are dot mentions found in here?
                #print(self.partner_ref_preds)
                #last_partner_reference = self.partner_ref_preds[-1].cpu().numpy()[:,0]
                # is the second dimension batch?
                #print(f"{self.name} reading")
                #print(last_partner_reference)
                #if not last_partner_reference.any() and self.actions and isinstance(self.actions[-1], Ask):
                    #response = ProductObservation({id: 0 for id in range(7)})
                    #d0 = TreeDebugger(agent.tree)
                    #print(d0)
                    #print(self.actions[-1])
                    #belief_update(
                        #agent, self.actions[-1], response,
                        #self.problem.env.state.object_states[agent.id],
                        #self.problem.env.state.object_states[agent.countdown_id],
                        #self.planner,
                    #)
                    #dd = TreeDebugger(agent.tree)
                    #print(dd)
                    #import pdb; pdb.set_trace()
                #else:
                    # PARTICULAR TO BETA-BERNOULLI
                    #merged_ref = []
                    #for idx in range(last_partner_reference.shape[0]):
                        #ref = last_partner_reference[idx]
                        # specificity requirement
                        #if ref.any() and ref.sum() <= 3:
                            #merged_ref.append(ref)
                    #if merged_ref:
                        #if not hasattr(agent, "tree") or agent.tree is None:
                            # need to plan to initialize tree
                            #plan(self.planner, self.problem, steps_left = 20 - self.timesteps)
                        #merged_ref = np.vstack(merged_ref).any(0)
                        #d0 = TreeDebugger(agent.tree)
                        #print(d0)
                        #for idx in merged_ref.nonzero()[0]:
                            # if they mentioned anything you have, do a belief update
                            # as if you asked about it
                            #array = np.zeros(7, dtype=np.int)
                            #array[idx.item()] = 1
                            #array[idx] = 1
                            #action = Ask(array)
                            #response = ProductObservation({id: array[id] for id in range(7)})
                            # check if unexpanded
                            #if agent.tree[action][response] is None:
                                #for _ in range(10):
                                    #self.planner.force_expansion(action, response)
                            #next_node = agent.tree[action][response]
                            # check if need particle rejuvenation
                            #num_particles = len(next_node.belief.particles)
                            #if num_particles == 0:
                                #for _ in range(10):
                                    #self.planner.force_expansion(action, response)
                            #belief_update(
                                #agent, action, response,
                                #self.problem.env.state.object_states[agent.id],
                                #self.problem.env.state.object_states[agent.countdown_id],
                                #self.planner,
                            #)
                            #dd = TreeDebugger(agent.tree)
                            #print(dd)
                            #import pdb; pdb.set_trace()
            # / POLICY BLOCK
        else:
            self.partner_ref_outs.append(None)
            self.partner_ref_preds.append(None)
        self.ref_outs.append(None)
        self.ref_preds.append(None)
        self.words.append(self.model.word2var(start_token).unsqueeze(0))
        self.words.append(Variable(inpt))
        if do_update:
            self.update_dot_h(ref_inpt=None, partner_ref_inpt=partner_ref_inpt,
                              num_markables=None, partner_num_markables=partner_num_markables,
                              ref_tgt=ref_tgt, partner_ref_tgt=partner_ref_tgt)
        if (self.selection_word_index == inpt).any():
            sel_idx = (self.selection_word_index == inpt.flatten()).nonzero()
            assert len(sel_idx) == 1
            # add one to offset from the start_token
            self.selection(sel_idx[0] + 1)
        if do_update:
            self.state = self.state._replace(turn=self.state.turn+1)
            self.timesteps += 1
            self.is_selection()
        #assert (torch.cat(self.words).size(0) == torch.cat(self.lang_hs).size(0))

    def rerank_language(self, outs, extra, dots_mentioned, dots_mentioned_per_ref, is_selection, can_confirm, generation_beliefs):
        target_num_markables = dots_mentioned_per_ref.size(1)
        #best_output = outputs[0][:lens[0]].unsqueeze(1)
        # return best_output

        num_candidates_before_filtering = len(extra['words'])

        all_markables, all_ref_boundaries = [], []
        all_ref_inpt, all_num_markables = [], []
        # all_ref_tgt = []

        indices_kept = []
        for ix, utterance in enumerate(extra['words']):
            # TODO: batch this
            markables, ref_boundaries = self.detect_markables(
                utterance, remove_pronouns=vars(self.model.args).get('dots_mentioned_no_pronouns', False)
            )
            ref_inpt, num_markables = self.markables_to_tensor(ref_boundaries)
            if num_markables.item() != target_num_markables:
                continue
            # ref_tgt = torch.zeros((1, num_markables.item(), 7)).long().to(self.device)
            all_markables.append(markables)
            all_ref_boundaries.append(ref_boundaries)
            all_ref_inpt.append(ref_inpt)
            all_num_markables.append(num_markables)
            # all_ref_tgt.append(ref_tgt)
            indices_kept.append(ix)
        if not all_markables:
            # no utterances with the target number of referents found, revert to the 1-best candidate
            this_extra = dict(
                **extra,
                was_rerankable=torch.zeros(num_candidates_before_filtering).bool(),
                ref_resolution_scores=torch.full((num_candidates_before_filtering,), -1e9),
                joint_scores=extra['output_logprobs'],
                language_model_scores=extra['output_logprobs'],
                chosen_index=torch.tensor(0).long(),
            )
            return outs, this_extra
        num_candidates = len(all_markables)
        ref_inpt = torch.cat(all_ref_inpt, dim=0)
        num_markables = torch.cat(all_num_markables, dim=0)
        # ref_tgt = torch.cat(all_ref_tgt)

        state_expanded = self.state.expand_bsz(num_candidates)
        dots_mentioned_expanded = dots_mentioned.expand(num_candidates, -1)
        dots_mentioned_per_ref_expanded = dots_mentioned_per_ref.expand(num_candidates, -1, -1)

        if is_selection is not None:
            raise NotImplementedError()
        if generation_beliefs is not None:
            raise NotImplementedError()

        prefix = self.model.word2var(YOU_TOKEN).unsqueeze(-1).expand(1, num_candidates)
        # T x num_candidates
        inpt = torch.cat((prefix, extra['outputs'][indices_kept].transpose(0,1)), dim=0)
        (reader_lang_hs, writer_lang_hs), new_state, _, _ = self.model._read(
            state_expanded, inpt, lens=extra['lens'][indices_kept]+1, pack=True,
            dots_mentioned=dots_mentioned_expanded,
            dots_mentioned_per_ref=dots_mentioned_per_ref_expanded,
            dots_mentioned_num_markables=num_markables,
            generation_beliefs=generation_beliefs,
            is_selection=is_selection,
            can_confirm=can_confirm,
        )
        ref_beliefs = state_expanded.make_beliefs('ref', self.timesteps, self.partner_ref_outs, self.ref_outs)
        ref_out = self.model.reference_resolution(
            new_state, reader_lang_hs, ref_inpt, num_markables,
            for_self=True, ref_beliefs=ref_beliefs
        )
        dots_mentioned_scores = models.reference_predictor.score_targets(ref_out, num_markables, dots_mentioned_per_ref_expanded)
        # ref_predictor = ReferencePredictor(self.args)
        # ref_loss, ref_pred, states = ref_predictor.forward(True, dots_mentioned_per_ref_expanded, ref_out, num_markables)
        # ref_pred_scores = models.reference_predictor.score_targets(ref_out, num_markables, ref_pred.transpose(0,1))
        weight = self.args.language_rerank_weight
        assert 0 <= weight <= 1.0

        dots_mentioned_scores_full = torch.full((num_candidates_before_filtering,), -1e9)
        dots_mentioned_scores_full[indices_kept] = dots_mentioned_scores

        candidate_scores = extra['output_logprobs'] * (1.0 - weight) + dots_mentioned_scores_full * weight
        candidate_index = candidate_scores.argmax()

        candidate_chosen = extra['outputs'][candidate_index]
        candidate_len = extra['lens'][candidate_index]

        candidate_kept = torch.zeros(num_candidates_before_filtering).bool()
        candidate_kept[indices_kept] = True

        this_extra = dict(
            **extra,
            was_rerankable=candidate_kept,
            ref_resolution_scores=dots_mentioned_scores_full,
            language_model_scores=extra['output_logprobs'],
            joint_scores=candidate_scores,
            chosen_index=candidate_index,
        )

        return candidate_chosen[:candidate_len].unsqueeze(1), this_extra

    def make_can_confirm(self, is_self, resolution_strategy='any'):
        if len(self.sents) == 0:
            can_confirm = make_can_confirm_single(len(self.sents), is_self, None, None, resolution_strategy=resolution_strategy)
        else:
            partner_ref_tgts = self.partner_ref_preds[-1]
            if partner_ref_tgts is not None:
                partner_ref_tgts = partner_ref_tgts.transpose(0,1)
            can_confirm = make_can_confirm_single(
                len(self.sents),
                is_self,
                torch.LongTensor([len(self.partner_markables[-1])]).to(is_self.device),
                partner_ref_tgts,
                resolution_strategy=resolution_strategy,
            )
        return can_confirm

    def write(
        self, max_words=100, force_words=None, detect_markables=True, start_token=YOU_TOKEN,
        dots_mentioned_per_ref_to_force=None,
        dots_mentioned_num_markables_to_force=None,
        ref_inpt=None,
        # used for oracle beliefs
        ref_tgt=None, partner_ref_tgt=None,
        is_selection=None,
        inference='sample',
        beam_size=1,
        sample_temperature_override=None,
        can_confirm=None,
        min_num_mentions=0,
        max_num_mentions=12,
        force_dots_mentioned=False,
        do_update = True,
    ):

        if can_confirm is None:
            is_self = torch.ones(1).bool()
            can_confirm = self.make_can_confirm(is_self, vars(self.model.args).get("confirmations_resolution_strategy", "any"))

        device = self.state.ctx.device

        if force_dots_mentioned:
            # JC: not used in selfplay, assume this is when testing ground truth?
            dots_mentioned_per_ref_candidates = [dots_mentioned_per_ref_to_force]
            #import pdb; pdb.set_trace()
        else:
            if not self.model.args.next_mention_prediction:
                dots_mentioned_per_ref_candidates = [None]
            else:
                if self.reader_lang_hs:
                    self.next_mention(lens=torch.LongTensor([self.reader_lang_hs[-1].size(0)]).to(self.device),
                                      dots_mentioned_num_markables_to_force=dots_mentioned_num_markables_to_force,
                                      min_num_mentions=min_num_mentions,
                                      max_num_mentions=max_num_mentions,
                                      can_confirm=can_confirm)
                else:
                    self.first_mention(dots_mentioned_num_markables_to_force=dots_mentioned_num_markables_to_force,
                                       min_num_mentions=min_num_mentions,
                                       max_num_mentions=max_num_mentions)

                if self.args.reranking_confidence:
                    assert self.args.language_rerank
                    assert self.args.next_mention_reranking
                    nm_candidates = self.next_mention_predictions_multi[-1]
                else:
                    nm_candidates = [self.next_mention_predictions[-1]]

                dots_mentioned_per_ref_candidates = [
                    t.transpose(0, 1) if t is not None else None
                    for t in nm_candidates
                ] if nm_candidates is not None else [None]

                #if self.policy == "beta_bernoulli":
                    # POLICY BLOCK
                    # JC: INSERT POLICY OUTPUT HERE, add another branch and force mentions
                    # that way we dont have to run prediction again...

                    #action = plan(self.planner, self.problem, steps_left = 20 - self.timesteps)
                    #print(action)
                    # TODO: SWITCH ON ACTION = SELECT
                    # OVERWRITE
                    #if isinstance(action, Ask):
                        #self.actions.append(action)
                        #dots_mentioned_per_ref_candidates = [
                            #torch.tensor(action.val, dtype=torch.int64).reshape((1, 1, 7))
                        #]
                    #elif isinstance(action, Select):
                        # TODO: FORCE SELECTION
                        #self.actions.append(action)
                        #z = torch.zeros(7, dtype=torch.int64).reshape((1, 1, 7))
                        #z[0,0,action.val] = 1
                        #dots_mentioned_per_ref_candidates = [z]
                    # / POLICY BLOCK

        best_generation_output = None
        best_generation_score = None

        if self.args.reranking_confidence:
            assert inference in ['beam', 'noised_beam', 'forgetful_noised_beam'] and self.args.language_rerank

        while best_generation_output is None:
            if self.args.DBG_GEN:
                self.dot_bags = []
            for dots_mentioned_per_ref in dots_mentioned_per_ref_candidates:
                if dots_mentioned_per_ref is None:
                    dots_mentioned_per_ref = torch.zeros((1, 0, 7)).bool().to(device)
                else:
                    dots_mentioned_per_ref = dots_mentioned_per_ref.bool()

                this_num_markables = torch.LongTensor([dots_mentioned_per_ref.size(1)]).to(device)
                dots_mentioned = dots_mentioned_per_ref.any(1)

                this_generation_score = None

                generation_beliefs = self.state.make_beliefs('generation', self.timesteps, self.partner_ref_outs, self.ref_outs)

                # MBP DEBUGGING
                def get_beam(dots_mentioned_per_ref):
                    import pdb; pdb.set_trace()
                    this_num_markables = torch.LongTensor([dots_mentioned_per_ref.size(1)]).to(device)
                    dots_mentioned = dots_mentioned_per_ref.any(1)
                    write_output_tpl = self.model.write_beam(
                        self.state, max_words, beam_size,
                        start_token=start_token,
                        dots_mentioned=dots_mentioned,
                        dots_mentioned_per_ref=dots_mentioned_per_ref,
                        dots_mentioned_num_markables=this_num_markables,
                        generation_beliefs=generation_beliefs,
                        is_selection=is_selection,
                        gumbel_noise=inference == 'noised_beam',
                        gumbel_noise_forgetful=inference == 'forgetful_noised_beam',
                        read_one_best=not self.args.language_rerank,
                        temperature=self.args.language_sample_temperature if inference in ['noised_beam', 'forgetful_noised_beam'] else 1.0,
                        keep_all_finished=self.args.language_beam_keep_all_finished,
                        can_confirm=can_confirm,
                    )
                    this_generation_output = GenerationOutput(
                        *((dots_mentioned_per_ref, this_num_markables) + write_output_tpl)
                    )
                    list_of_outputs = [
                        " ".join(sentence)
                        for sentence in write_output_tpl[-1]["words"]
                    ]
                    [print(x) for x in list_of_outputs]

                #if False and self.args.DBG_GEN:
                if self.args.DBG_GEN:
                    # for example S_pGlR0nKz9pQ4ZWsw, construct triangle mention
                    x = torch.zeros(1, 4, 7, dtype=bool, device = 0)
                    x[0,0,2:4] = 1
                    x[0,1,2] = 1
                    x[0,2,3] = 1
                    x[0,3,4] = 1
                    get_beam(x)
                    # 1 x num_mentions x 7
                    print("debug example")
                    import pdb; pdb.set_trace()
                # / MBP DEBUGGING

                if inference == 'sample':
                    sample_temperature = sample_temperature_override if sample_temperature_override is not None else self.args.temperature
                    write_output_tpl = self.model.write(
                        self.state, max_words, sample_temperature,
                        start_token=start_token,
                        force_words=force_words,
                        dots_mentioned=dots_mentioned,
                        dots_mentioned_per_ref=dots_mentioned_per_ref,
                        dots_mentioned_num_markables=this_num_markables,
                        generation_beliefs=generation_beliefs,
                        is_selection=is_selection,
                        can_confirm=can_confirm,
                    )
                    assert len(dots_mentioned_per_ref_candidates) == 1
                    this_generation_output = GenerationOutput(
                        *((dots_mentioned_per_ref, this_num_markables) + write_output_tpl)
                    )
                    best_generation_output = this_generation_output
                    break
                elif inference in ['beam', 'noised_beam', 'forgetful_noised_beam']:
                    assert not force_words
                    write_output_tpl = self.model.write_beam(
                        self.state, max_words, beam_size,
                        start_token=start_token,
                        dots_mentioned=dots_mentioned,
                        dots_mentioned_per_ref=dots_mentioned_per_ref,
                        dots_mentioned_num_markables=this_num_markables,
                        generation_beliefs=generation_beliefs,
                        is_selection=is_selection,
                        gumbel_noise=inference == 'noised_beam',
                        gumbel_noise_forgetful=inference == 'forgetful_noised_beam',
                        read_one_best=not self.args.language_rerank,
                        temperature=self.args.language_sample_temperature if inference in ['noised_beam', 'forgetful_noised_beam'] else 1.0,
                        keep_all_finished=self.args.language_beam_keep_all_finished,
                        can_confirm=can_confirm,
                    )
                    this_generation_output = GenerationOutput(
                        *((dots_mentioned_per_ref, this_num_markables) + write_output_tpl)
                    )
                    # MBP DBG
                    if self.args.DBG_GEN:
                        self.dot_bags.append(models.utils.bit_to_int_array(dots_mentioned).item())
                        list_of_outputs = [
                            " ".join(sentence)
                            for sentence in write_output_tpl[-1]["words"]
                        ]
                        [print(x) for x in list_of_outputs]
                        print(dots_mentioned_per_ref.int())
                        print(dots_mentioned_per_ref.any(-1).int())
                        print("actual candidates")
                        import pdb; pdb.set_trace()
                    # / MBP DBG
                    if self.args.language_rerank:
                        self.decoded_beam_best.append(self._decode(this_generation_output.outs, self.model.word_dict))
                        if dots_mentioned_per_ref is None or dots_mentioned_per_ref.size(1) == 0:
                            # outs = outs
                            this_generation_score = None
                        else:
                            outs, extra = self.rerank_language(
                                this_generation_output.outs, this_generation_output.extra,
                                dots_mentioned, dots_mentioned_per_ref, is_selection, can_confirm,
                                generation_beliefs,
                            )
                            this_generation_output = this_generation_output._replace(outs=outs, extra=extra)
                            if self.args.reranking_confidence and len(dots_mentioned_per_ref_candidates) > 1:
                                if 'ref_resolution_scores' not in extra:
                                    # no utterances with the target number of mentions found in the beam!
                                    this_generation_score = None
                                else:
                                    joint_score = extra['joint_scores'][extra['chosen_index']]
                                    ref_resolution_score = extra['ref_resolution_scores'][extra['chosen_index']]

                                    this_generation_score = {
                                        'l0': ref_resolution_score,
                                        'joint': joint_score,
                                    }[self.args.reranking_confidence_score]
                        reader_and_writer_lang_hs, state = self.model.read(
                            self.state, this_generation_output.outs,
                            dots_mentioned=dots_mentioned,
                            dots_mentioned_per_ref=dots_mentioned_per_ref,
                            dots_mentioned_num_markables=this_num_markables,
                            prefix_token=start_token,
                            is_selection=is_selection,
                            can_confirm=can_confirm,
                        )
                        this_generation_output = this_generation_output._replace(
                            reader_and_writer_lang_hs=reader_and_writer_lang_hs,
                            state=state
                        )
                else:
                    raise NotImplementedError(f"inference = {inference}")
                if self.args.reranking_confidence and len(dots_mentioned_per_ref_candidates) > 1:
                    if this_generation_score is not None:
                        if best_generation_score is None or this_generation_score > best_generation_score:
                            best_generation_output = this_generation_output
                            best_generation_score = this_generation_score
                        if self.args.reranking_confidence_type == 'first_above_threshold' and this_generation_score.exp().item() >= self.args.reranking_confidence_exp_score_threshold:
                            assert best_generation_output is not None
                            break
                else:
                    best_generation_output = this_generation_output
                    break
            # FOR MEASURING NUM UNIQUE DOT BAGS
            if self.args.DBG_GEN:
                print(f"DBG: uniq {len(set(self.dot_bags))} out of {len(self.dot_bags)}")
            if best_generation_output is None:
                dots_mentioned_per_ref_candidates = [self.next_mention_predictions[-1]]
                dots_mentioned_per_ref_candidates = [
                    t.transpose(0, 1) if t is not None else None
                    for t in dots_mentioned_per_ref_candidates
                ]

        assert best_generation_output is not None
        self.dots_mentioned_per_ref_chosen.append(best_generation_output.dots_mentioned_per_ref)

        if do_update and best_generation_output.logprobs is not None:
            self.logprobs.extend(best_generation_output.logprobs)

        self.reader_lang_hs.append(best_generation_output.reader_and_writer_lang_hs[0])
        self.writer_lang_hs.append(best_generation_output.reader_and_writer_lang_hs[1])

        if do_update: self.state = best_generation_output.state

        if detect_markables:
            assert self.markable_detector is not None
            markables, ref_boundaries = self.detect_markables(self._decode(best_generation_output.outs, self.model.word_dict), remove_pronouns=False)
            ref_inpt, num_markables = self.markables_to_tensor(ref_boundaries)
            if self.model.args.dot_recurrence_oracle and best_generation_output.dots_mentioned_per_ref.size(1) == num_markables.item():
                assert self.model.args.dot_recurrence_oracle_for == ['self']
                ref_tgt = best_generation_output.dots_mentioned_per_ref
            else:
                ref_tgt = torch.zeros((1, num_markables.item(), 7)).long().to(self.device)

            self.ref_inpts.append(ref_inpt)
            self.markables.append(markables)
            self.partner_ref_inpts.append(None)
            self.partner_markables.append([])

        if ref_inpt is not None and num_markables is not None:
            self.predict_referents(ref_inpt, num_markables)
        else:
            self.ref_outs.append(None)
            self.ref_preds.append(None)
        self.partner_ref_outs.append(None)
        self.partner_ref_preds.append(None)
        #self.words.append(self.model.word2var('YOU:').unsqueeze(0))
        self.words.append(best_generation_output.outs)
        self.sents.append(torch.cat([self.model.word2var(start_token).unsqueeze(1), best_generation_output.outs], 0))
        self.extras.append(best_generation_output.extra)
        extra = best_generation_output.extra
        if 'joint_scores' in extra:
            self.joint_scores.append(extra['joint_scores'][extra['chosen_index']].item())
            self.ref_resolution_scores.append(extra['ref_resolution_scores'][extra['chosen_index']].item())
            self.language_model_scores.append(extra['output_logprobs'][extra['chosen_index']].item())
        # JC: doesnt look like the next mention candidate is used here
        if do_update:
            self.update_dot_h(
                ref_inpt=ref_inpt, partner_ref_inpt=None,
                num_markables=num_markables, partner_num_markables=None,
                ref_tgt=ref_tgt, partner_ref_tgt=partner_ref_tgt,
            )
        # partner next mentions;
        # we probably don't actually need to do this
        # self.next_mention(lens=torch.LongTensor([reader_lang_hs.size(0)]).to(self.device),
        #                   num_markables_to_force=None,
        #                   )
        if (self.selection_word_index == best_generation_output.outs).any():
            sel_idx = (self.selection_word_index == best_generation_output.outs.flatten()).nonzero()
            assert len(sel_idx) == 1
            # add one to offset from the start_token
            self.selection(sel_idx[0] + 1)
        if do_update:
            self.state = self.state._replace(turn=self.state.turn+1)
            self.timesteps += 1
            self.is_selection()

        """if self.args.visualize_referents:
            #utterance = self._decode(outs, self.model.word_dict)[1:-1]
            #const_tree = list(self.corenlp_parser.parse(utterance))
            utterance = self._decode(outs, self.model.word_dict)
            ref_inpt = [3, 6, len(utterance) - 1]
            ref_inpt = torch.Tensor(ref_inpt).long().unsqueeze(0).unsqueeze(0)
            ref_out = self.model.reference_resolution(self.ctx_h, lang_hs.unsqueeze(1), ref_inpt)
            pdb.set_trace()"""

        #if not (torch.cat(self.words).size(0) + 1 == torch.cat(self.lang_hs).size(0)):
        #    pdb.set_trace()
        #assert (torch.cat(self.words).size(0) + 1 == torch.cat(self.lang_hs).size(0))
        # remove 'YOU:'
        # outs = outs.narrow(0, 1, outs.size(0) - 1)
        
        # FOR DEBUGGING
        #meh = self._decode(best_generation_output.outs, self.model.word_dict)
        #import pdb; pdb.set_trace()
        return self._decode(best_generation_output.outs, self.model.word_dict)

    def next_mention(self, lens, dots_mentioned_num_markables_to_force=None, min_num_mentions=0, max_num_mentions=12, can_confirm=None):
        mention_beliefs = self.state.make_beliefs(
            'mention', self.timesteps-1, self.partner_ref_outs, self.ref_outs,
        )
        mention_latent_beliefs = self.state.make_beliefs(
            'next_mention_latents', self.timesteps-1, self.partner_ref_outs, self.ref_outs,
        )
        if self.args.next_mention_reranking and self.args.next_mention_candidate_generation == 'topk_multi_mention':
            assert dots_mentioned_num_markables_to_force is None
            dots_mentioned_num_markables_to_force = torch.Tensor([self.args.next_mention_reranking_max_mentions]).to(self.device)
        next_mention_latents = self.model.next_mention_latents(
            self.state, self.writer_lang_hs[-1], lens, mention_beliefs,
            mention_latent_beliefs,
            dots_mentioned_num_markables_to_force=dots_mentioned_num_markables_to_force,
            min_num_mentions=min_num_mentions,
            max_num_mentions=max_num_mentions,
            can_confirm=can_confirm,
        )
        self.next_mention_latents.append(next_mention_latents)
        nm_preds, nm_cands, nm_preds_multi, nm_indices_sorted = self.next_mention_prediction_and_candidates_from_latents(self.state, next_mention_latents)
        self.next_mention_candidates.append(nm_cands)
        self.next_mention_predictions.append(nm_preds)
        self.next_mention_predictions_multi.append(nm_preds_multi)
        self.next_mention_indices_sorted.append(nm_indices_sorted)
        return nm_preds

    def first_mention(self, dots_mentioned_num_markables_to_force=None, min_num_mentions=0, max_num_mentions=12):
        if self.args.next_mention_reranking and self.args.next_mention_candidate_generation == 'topk_multi_mention':
            assert dots_mentioned_num_markables_to_force is None
            dots_mentioned_num_markables_to_force = torch.Tensor([self.args.next_mention_reranking_max_mentions]).to(self.device)
        next_mention_latents = self.model.first_mention_latents(
            self.state,
            dots_mentioned_num_markables=dots_mentioned_num_markables_to_force,
            force_next_mention_num_markables=dots_mentioned_num_markables_to_force is not None,
            min_num_mentions=min_num_mentions,
            max_num_mentions=max_num_mentions,
        )
        self.next_mention_latents.append(next_mention_latents)
        nm_preds, nm_cands, nm_preds_multi, nm_indices_sorted = self.next_mention_prediction_and_candidates_from_latents(self.state, next_mention_latents)
        self.next_mention_candidates.append(nm_cands)
        self.next_mention_predictions.append(nm_preds)
        self.next_mention_predictions_multi.append(nm_preds_multi)
        self.next_mention_indices_sorted.append(nm_indices_sorted)
        return nm_preds

    def is_selection(self):
        self.is_selection_outs.append(self.model.is_selection_prediction(self.state))

    def selection(self, sel_idx):
        selection_beliefs = self.state.make_beliefs(
            'selection', self.timesteps, self.partner_ref_outs, self.ref_outs,
        )
        sel_out = self.model.selection(self.state, self.reader_lang_hs[-1], sel_idx, beliefs=selection_beliefs)
        self.sel_outs.append(sel_out)
        return sel_out

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def _choose(self, sample=False):
        # outs_emb = torch.cat(self.reader_lang_hs).unsqueeze(1)
        # sel_idx = torch.Tensor(1).fill_(outs_emb.size(0) - 1).long()
        # choice_logit = self.model.selection(self.state.ctx_differences, self.state.ctx_h, outs_emb, sel_idx)
        if not self.sel_outs:
            # force selection using the last utterance
            sel_idx = self.reader_lang_hs[-1].size(0) - 1
            self.selection(torch.tensor([sel_idx]).long().to(self.device))

        choice_logit, _, _ = self.sel_outs[-1]

        prob = F.softmax(choice_logit, dim=-1)
        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=-1).gather(1, idx)
        else:
            _, idx = prob.max(1, keepdim=True)
            logprob = None

        # Pick only your choice
        return idx.item(), prob.gather(1, idx), logprob

    def choose(self):
        if self.args.eps < np.random.rand():
            choice, _, _ = self._choose(sample=False)
        else:
            choice, _, logprob = self._choose(sample=True)
            self.logprobs.append(logprob)

        choice, _, _ = self._choose()
        if hasattr(self, "real_ids") and self.real_ids:
            choice = self.real_ids[choice]
        return choice

    def update(self, agree, reward, choice=None):
        if not self.train:
            return

        self.t += 1
        if len(self.logprobs) == 0:
            return

        self.all_rewards.append(reward)

        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        if self.args.visual and self.t % 10 == 0:
            self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.agree_reward_plot.update('reward', self.t, reward_agree)
            self.agree_reward_plot.update('partner_reward', self.t, partner_reward_agree)
            self.loss_plot.update('loss', self.t, loss.data[0][0])

        self.opt.step()
