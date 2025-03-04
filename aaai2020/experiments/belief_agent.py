
import numpy as np
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
)

from agent import RnnAgent, YOU_TOKEN, THEM_TOKEN, GenerationOutput
from belief import (
    AndOrBelief, OrAndBelief, OrBelief, OrAndOrBelief,
    ConfigBelief,
    entropy, expand_plan,
)
from cog_belief import CostBelief, EgoCostBelief

from template_rec import render, render_select

from hfutils import DescriptionFormat
from hfdata import describe_plan_specific_dots, describe_plan_sparse

NO_RESPONSE = 0
CONFIRM = 1
DISCONFIRM = 2

class BeliefAgent(RnnAgent):

    @staticmethod
    def add_args(parser):
        RnnAgent.add_args(parser)
        parser.add_argument(
            "--belief",
            choices = [
                "or", "and", "andor", "orand", "orandor",
                "or_egocentric",
                "cost", "cost_egocentric",
            ],
            default = "or",
            help = "Partner response model",
        )
        parser.add_argument(
            "--belief_entropy_threshold",
            type = float,
            default = 2.,
            help = "Belief entropy threshold for selection heuristic",
        )
        parser.add_argument(
            "--belief_threshold",
            type = float,
            default = 0.8,
            help = "Belief probability hreshold for selection",
        )
        parser.add_argument(
            "--absolute_bucketing",
            choices = [0,1],
            default = 1,
            type=int,
            help = "If on=1, switch from relative bucketing to absolute bucketing of unary features size/color",
        )
        parser.add_argument(
            "--num_size_buckets",
            default = 5,
            type=int,
            help = "Number of size buckets for dots (large, small, etc). more is finer grained",
        )
        parser.add_argument(
            "--num_color_buckets",
            default = 5,
            type=int,
            help = "Number of color buckets for dots (dark, light, etc). more is finer grained",
        )
        parser.add_argument(
            "--length_coef",
            default = 0,
            type = float,
            help = "length coef in utility",
        )
        parser.add_argument(
            "--diameter_coef",
            default = 0,
            type = float,
            help = "plan diameter (max dist b/w 2 points) coef in utility",
        )
        parser.add_argument(
            "--contiguity_coef",
            default = 0,
            type = float,
            help = "plan contiguity coef in utility",
        )
        parser.add_argument(
            "--select_config_size",
            default = 3,
            type = float,
            help = "config size for communicating selection",
        )

    # same init as RnnAgent, initialize belief in feed_context
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        # NEED TO ADJUST THIS, pass in as argument
        #response_pretrained_path = "models/save_pretrained"
        #self.tokenizer = AutoTokenizer.from_pretrained(response_pretrained_path)
        #self.confirmation_predictor = AutoModelForSequenceClassification.from_pretrained(
            #response_pretrained_path)
        assert self.confirmation_tokenizer is not None
        assert self.confirmation_predictor is not None

        # for selection heuristic based on entropy (DEPRECATED)
        self.entropy_threshold = self.args.belief_entropy_threshold
        self.belief_threshold = self.args.belief_threshold

        # utility coefficients
        self.length_coef = self.args.length_coef
        self.diameter_coef = self.args.diameter_coef
        self.contiguity_coef = self.args.contiguity_coef

        # select config size
        self.select_config_size = self.args.select_config_size

    def feed_context(self, context, belief_constructor):
        super().feed_context(context, belief_constructor)
        self.num_dots = 7
        if self.args.belief == "or":
            self.belief = OrBelief(
                self.num_dots, context,
                absolute = self.args.absolute_bucketing == 1,
                num_size_buckets = self.args.num_size_buckets,
                num_color_buckets = self.args.num_color_buckets,
                use_diameter = self.diameter_coef > 0,
                use_contiguity = self.contiguity_coef > 0,
            )
        elif self.args.belief == "or_egocentric":
            self.belief = ConfigBelief(
                self.num_dots, context,
                absolute = self.args.absolute_bucketing == 1,
                num_size_buckets = self.args.num_size_buckets,
                num_color_buckets = self.args.num_color_buckets,
                use_diameter = self.diameter_coef > 0,
                use_contiguity = self.contiguity_coef > 0,
            )
        elif self.args.belief == "cost":
            self.belief = CostBelief(
                self.num_dots, context,
                absolute = self.args.absolute_bucketing == 1,
                num_size_buckets = self.args.num_size_buckets,
                num_color_buckets = self.args.num_color_buckets,
                use_diameter = self.diameter_coef > 0,
                use_contiguity = self.contiguity_coef > 0,
            )
        elif self.args.belief == "cost_egocentric":
            self.belief = EgoCostBelief(
                self.num_dots, context,
                absolute = self.args.absolute_bucketing == 1,
                num_size_buckets = self.args.num_size_buckets,
                num_color_buckets = self.args.num_color_buckets,
                use_diameter = self.diameter_coef > 0,
                use_contiguity = self.contiguity_coef > 0,
            )
        elif self.args.belief == "and":
            self.belief = AndBelief(self.num_dots, context)
        elif self.args.belief == "andor":
            self.belief = AndOrBelief(self.num_dots, context)
        elif self.args.belief == "orand":
            self.belief = OrBelief(self.num_dots, context)
        elif self.args.belief == "orandor":
            self.belief = OrAndOrBelief(self.num_dots, context)
        else:
            raise ValueError

        self.prior = self.belief.prior
        # per-turn accumulators
        self.next_mention_plans = []
        self.next_confirms = []
        self.responses = []
        self.response_logits = []
        self.side_infos = []
        # symbolic accumulators
        #
        # selection index
        self.sel_idx = None
        self.sel_config = None

        # initialize dot description for BART
        #import pdb; pdb.set_trace()
        # maybe not necessary


    # do not overload super().is_selection
    def should_select(self):
        # should we select?
        # uniform entropy over 128 is 4.85
        #print(np.round(self.belief.marginals(self.prior), 2))
        #print(entropy(self.prior))
        max_belief = self.belief.marginals(self.prior).max()
        select = max_belief > self.belief_threshold
        #H = entropy(self.prior)
        #select = H < self.entropy_threshold
        return select

    def select_dot(self):
        # chooses the most likely shared configuration and a dot within it
        # maybe should choose a dot and the 2 nearest shared?

        #config = self.belief.marginal_size(self.prior, 3)
        #config = self.belief.marginal_size(self.prior, 4)
        config = self.belief.marginal_size(self.prior, self.select_config_size)

        marginals = self.belief.marginals(self.prior)
        marginals[~config.astype(bool)] = 0
        index = marginals.argmax()
        # index is absolute index in self.real_ids

        self.sel_config = config
        self.sel_idx = index

        # relative index within config
        rel_idx = marginals[config.astype(bool)].argmax()

        return config, index, rel_idx


    def read(
        self, inpt_words, *args, **kwargs,
    ):
        # call parent's read method
        super().read(inpt_words, *args, **kwargs)

        # TODO: DOES INPT_WORDS HAVE THEM?
        # run confirmation prediction
        # inpt_words: [String] are partner's last utterance
        text = " ".join(inpt_words)
        # raw text strings for BART are tracked in agent.py:read
        tokenized_text = self.confirmation_tokenizer(text)
        response_struct = self.confirmation_predictor(
            torch.tensor(tokenized_text["input_ids"])[None],
        )
        response_logits = response_struct.logits[0].log_softmax(-1)
        label = response_logits.argmax().item()

        plan = self.next_mention_plans[-1] if self.next_mention_plans else None
        partner_mentions_per_ref = self.partner_ref_preds[-1]

        # response belief update
        if label != NO_RESPONSE and plan is not None:
            response = 1 if label == CONFIRM else 0
            self.prior = self.belief.posterior(self.prior, plan, response)
        # side info belief update
        side_info = None
        if partner_mentions_per_ref is not None:
            print("Previous partner mentions")
            print(partner_mentions_per_ref)
            # WARNING: mention detector is really bad
            # TODO: not ideal. prefer to MARGINALIZE over partner mentions
            # UNCERTAINTY IS IMPORTANT HERE
            # loop over partner_mentions_per_ref, or assume collapsed?
            partner_mentions_collapsed = (
                partner_mentions_per_ref.any(0)[0].cpu().numpy().astype(int)
            )
            # partner_mentions_collapsed - plan
            side_info = (
                ((partner_mentions_collapsed - plan) > 0).astype(int)
                if plan is not None
                else partner_mentions_collapsed
            )
            if side_info is not None and side_info.any():
                self.prior = self.belief.posterior(self.prior, side_info, 1)

        # add None to next_mention_plans, since reading turn => no generation
        self.next_mention_plans.append(None)
        self.responses.append(label)
        self.response_logits.append(response_logits.tolist())
        self.side_infos.append(side_info)


    def write_supervised(self, force_words=None, *args, **kwargs):
        dots_mentioned_per_ref_to_force = None
        # if write is given forced_words, then throw away plan. otherwise
        plan = None
        if force_words is None:
            # generate next mention plan
            utilities = self.belief.compute_utilities(
                self.prior,
                length_coef = self.length_coef,
                diameter_coef = self.diameter_coef,
                contiguity_coef = self.contiguity_coef,
            )
            cs, us = self.belief.viz_belief(utilities, n=4)
            # TODO: MULTIPLE PLANS
            plan = cs[0]
            # convert plan to dots_mentioned
            dots_mentioned = torch.tensor(expand_plan(
                plan,
                unroll=self.model.args.next_mention_prediction_type == "multi_reference",
            ))
            # heuristic: produce next mentions per ref
            #dots_mentioned_per_ref_to_force = [dots_mentioned]
            # dots_mentioned_per_ref_to_force: num_mentions x bsz=1 x num_dots=7
            dots_mentioned_per_ref_to_force = dots_mentioned.transpose(0,1)

            # TODO: fix dot description generation, load correct args
            dot_description, _ = describe_plan_specific_dots(
                self.context,
                plan,
                # URGENT TODO: NEED TO CHANGE THIS DYNAMICALLY
                use_unordered_pairwise = False,
                format = DescriptionFormat.SrcRelsTgt,
                #format = DescriptionFormat.SrcRelTgts,
            )
            previous_text = " ".join(w for sent in self.text_history for w in sent)
            plan_description = describe_plan_sparse(plan)
            # HARD CODE no confirmation / selection
            generator_input = (
                f"{dot_description} "
                f"[MSEP] {previous_text} "
                "[MSEP] confirmation: none "
                "[MSEP] should we select? no "
                "[MSEP] selection: not yet "
                f"[MSEP] {plan_description}"
            ) if self.language_generator.generate_given_text else (
                f"{dot_description} "
                "[MSEP] confirmation: none "
                "[MSEP] should we select? no "
                "[MSEP] selection: not yet "
                f"[MSEP] {plan_description}"
            )
            outputs = self.language_generator.generate(
                self.generation_tokenizer(generator_input, return_tensors="pt")["input_ids"],
                num_beams = 16,
                num_return_sequences = 16,
                output_scores = True,
                return_dict_in_generate = True,
                max_new_tokens = 80,
            )
            sentence_candidates = self.generation_tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True)
            lm_scores = outputs.sequences_scores
            # SAVE FOR LOGGING ONLY
            self.bart_input = generator_input
            self.bart_sentence_candidates = sentence_candidates
            self.bart_lm_scores = lm_scores

        output = super().write(
            force_dots_mentioned = dots_mentioned_per_ref_to_force is not None,
            dots_mentioned_per_ref_to_force = dots_mentioned_per_ref_to_force,
            force_words = force_words,
            *args, **kwargs,
        )

        if force_words is not None:
            # set plan to the resolved refs of the forced utt
            plan = (self.ref_preds[-1].any(0)[0].cpu().int().numpy()
                if self.ref_preds[-1] is not None
                else None
            )

        # add plan to next_mention_plan history
        self.next_mention_plans.append(plan)
        self.responses.append(None)
        self.response_logits.append(None)
        self.side_infos.append(None)


        return output


    def read_symbolic(
        self,
        confirm=None,
        mention_features=None,
        select_features = None,
        select_rel_idx = None,
    ):
        # process confirmation
        if confirm is not None and len(self.next_mention_plans) > 0:
            plan = self.next_mention_plans[-1] if self.next_mention_plans else None
            self.prior = self.belief.posterior(self.prior, plan, confirm)

        # process mentions
        # returns combinations that match
        mentions = self.belief.resolve_utt(*mention_features)
        if len(mentions) > 0:
            # need to create plan and nonzero it
            mention_plan = np.zeros(self.num_dots, dtype=int)
            mention_plan[mentions[0]] = 1
            self.prior = self.belief.posterior(self.prior, mention_plan, 1)

        # add None to next_mention_plans, since reading turn => no generation
        self.next_mention_plans.append(None)
        self.next_confirms.append(None)
        self.responses.append(confirm)
        self.response_logits.append(None)
        self.side_infos.append(mention_features)

        self.state = self.state._replace(turn=self.state.turn+1)

        if select_features is not None:
            # writer has selected
            select_configs = self.belief.resolve_utt(*select_features)#[0]
            if len(select_configs) > 1:
                print("MULTIPLE SELECTION CONFIGS")
            if len(select_configs) > 0:
                # if able to resolve features
                # otherwise just guess
                self.sel_config = select_configs[0]
                self.sel_idx = select_configs[0][select_rel_idx]


    def write_symbolic(self):
        # selection happens during writing: output <selection>
        should_select = self.should_select()
        select_feats = None
        select_rel_idx = None
        if should_select:
            # switch modes to selection communication
            # which dot to select?
            select_config, select_idx, select_rel_idx = self.select_dot()
            # communicate a dot configuration and index of dot
            select_feats = self.belief.get_feats(select_config)

        # generate next mention plan
        utilities = self.belief.compute_utilities(
            self.prior,
            length_coef = self.length_coef,
            diameter_coef = self.diameter_coef,
            contiguity_coef = self.contiguity_coef,
        )
        cs, us = self.belief.viz_belief(utilities, n=4)
        plan = cs[0]
        feats = self.belief.get_feats(plan)

        confirm = None
        if len(self.side_infos) > 0 and self.side_infos[-1] is not None:
            partner_mention = self.side_infos[-1]
            resolved = self.belief.resolve_utt(*partner_mention)
            confirm = int(len(resolved) > 0)

        # add plan to next_mention_plan history
        self.next_mention_plans.append(plan)
        self.next_confirms.append(confirm)
        self.responses.append(None)
        self.response_logits.append(None)
        self.side_infos.append(None)

        self.state = self.state._replace(turn=self.state.turn+1)

        return confirm, feats, select_feats, select_rel_idx


    def choose(self):
        if self.sel_idx is None:
            self.select_dot()
        #return self.real_ids[self.sel_idx]
        return self.sel_idx

    def write(self, *args, **kwargs):
        # SYMBOLIC PORTION
        # selection happens during writing: output <selection>
        should_select = self.should_select()
        select_config = None
        select_feats = None
        select_idx = None
        select_rel_idx = None
        if should_select:
            # switch modes to selection communication
            # which dot to select?
            select_config, select_idx, select_rel_idx = self.select_dot()
            # communicate a dot configuration and index of dot
            select_feats = self.belief.get_feats(select_config)
            select_ids = select_config.nonzero()[0]

        # generate next mention plan
        utilities = self.belief.compute_utilities(
            self.prior,
            length_coef = self.length_coef,
            diameter_coef = self.diameter_coef,
            contiguity_coef = self.contiguity_coef,
        )
        cs, us = self.belief.viz_belief(utilities, n=4)
        plan = cs[0]
        feats = self.belief.get_feats(plan)
        ids = plan.nonzero()[0]

        confirm = None
        if len(self.side_infos) > 0 and self.side_infos[-1] is not None:
            partner_mention = self.side_infos[-1]
            resolved = partner_mention.any()
            confirm = int(resolved)
            #print(f"RESOLVED: {resolved}")

        # add plan to next_mention_plan history
        self.next_mention_plans.append(plan)
        self.next_confirms.append(confirm)
        self.responses.append(None)
        self.response_logits.append(None)
        self.side_infos.append(None)

        #self.state = self.state._replace(turn=self.state.turn+1)

        #return confirm, feats, select_feats, select_rel_idx
        # WORDS
        force_words = render(*feats, ids, confirm=confirm) if not should_select else (
            render_select(
                select_feats,
                select_idx,
                select_ids,
                confirm = confirm,
            )
        )
        #print(plan)
        #print(select_config)
        #print(select_idx)
        #print(force_words)
        if should_select:
            force_words_split = [force_words.split() + ["<selection>"]]
        else:
            force_words_split = [force_words.split() + ["<eos>"]]
        #import pdb; pdb.set_trace()

        # FEED TO NEURAL
        # convert plan to dots_mentioned
        dots_mentioned = torch.tensor(expand_plan(
            plan,
            unroll=self.model.args.next_mention_prediction_type == "multi_reference",
        ))
        # heuristic: produce next mentions per ref
        #dots_mentioned_per_ref_to_force = [dots_mentioned]
        # dots_mentioned_per_ref_to_force: num_mentions x bsz=1 x num_dots=7
        dots_mentioned_per_ref_to_force = dots_mentioned.transpose(0,1)

        output = super().write(
            force_dots_mentioned = True,
            dots_mentioned_per_ref_to_force = dots_mentioned_per_ref_to_force,
            force_words = force_words_split,
            *args, **kwargs,
        )

        # ref_preds?!

        """
        if force_words is not None:
            # set plan to the resolved refs of the forced utt
            plan = (self.ref_preds[-1].any(0)[0].cpu().int().numpy()
                if self.ref_preds[-1] is not None
                else None
            )

        # add plan to next_mention_plan history
        self.next_mention_plans.append(plan)
        self.responses.append(None)
        self.response_logits.append(None)
        self.side_infos.append(None)
        """

        #import pdb; pdb.set_trace()

        # write takes artificially batched words. unbatch
        return force_words_split[0]
