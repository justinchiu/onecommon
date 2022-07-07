
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

from template_rec import render, render_select

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
                "egocentric",
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
            "--absolute_bucketing",
            choices = [0,1],
            default = 1,
            type=int,
            help = "If on=1, switch from relative bucketing to absolute bucketing of unary features size/color",
        )
        parser.add_argument(
            "--length_coef",
            default = 0,
            type = float,
            help = "length coef in utility",
        )

    # same init as RnnAgent, initialize belief in feed_context
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        # NEED TO ADJUST THIS, pass in as argument
        #response_pretrained_path = "models/save_pretrained"
        #self.tokenizer = AutoTokenizer.from_pretrained(response_pretrained_path)
        #self.confirmation_predictor = AutoModelForSequenceClassification.from_pretrained(
            #response_pretrained_path)
        assert self.tokenizer is not None
        assert self.confirmation_predictor is not None

        # for selection heuristic based on entropy
        self.entropy_threshold = self.args.belief_entropy_threshold

        # utility coefficients
        self.length_coef = self.args.length_coef

    def feed_context(self, context, belief_constructor):
        super().feed_context(context, belief_constructor)
        self.num_dots = 7
        if self.args.belief == "or":
            self.belief = OrBelief(
                self.num_dots, context,
                absolute = self.args.absolute_bucketing == 1,
            )
        elif self.args.belief == "egocentric":
            self.belief = ConfigBelief(
                self.num_dots, context,
                absolute = self.args.absolute_bucketing == 1,
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

    # do not overload super().is_selection
    def should_select(self):
        # should we select?
        # uniform entropy over 128 is 4.85
        #print(np.round(self.belief.marginals(self.prior), 2))
        #print(entropy(self.prior))
        H = entropy(self.prior)
        select = H < self.entropy_threshold
        return select

    def select_dot(self):
        # chooses the most likely shared configuration and a dot within it
        # maybe should choose a dot and the 2 nearest shared?

        #config = self.belief.marginal_size(self.prior, 3)
        config = self.belief.marginal_size(self.prior, 4)

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

        # run confirmation prediction
        # inpt_words: [String] are partner's last utterance
        text = " ".join(inpt_words)
        tokenized_text = self.tokenizer(text)
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
            )
            cs, us = self.belief.viz_belief(utilities, n=4)
            # TODO: MULTIPLE PLANS
            plan = cs[0]
            # convert plan to dots_mentioned
            dots_mentioned = torch.tensor(expand_plan(plan))
            # heuristic: produce next mentions per ref
            #dots_mentioned_per_ref_to_force = [dots_mentioned]
            # dots_mentioned_per_ref_to_force: num_mentions x bsz=1 x num_dots=7
            dots_mentioned_per_ref_to_force = dots_mentioned.transpose(0,1)

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
        # selection happens during writing: output <select>
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
        # selection happens during writing: output <select>
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
        )
        cs, us = self.belief.viz_belief(utilities, n=4)
        plan = cs[0]
        feats = self.belief.get_feats(plan)

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
        force_words = render(*feats, confirm=confirm) if not should_select else (
            render_select(select_feats, select_rel_idx, confirm=confirm)
        )
        force_words_split = [force_words.split() + ["<eos>"]]

        # FEED TO NEURAL
        # convert plan to dots_mentioned
        dots_mentioned = torch.tensor(expand_plan(plan))
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
