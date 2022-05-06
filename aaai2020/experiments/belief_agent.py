
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
)

from agent import RnnAgent, YOU_TOKEN, THEM_TOKEN, GenerationOutput
from belief import (
    AndOrBelief, OrAndBelief, OrBelief, OrAndOrBelief,
    process_ctx, expand_plan,
)

NO_RESPONSE = 0
CONFIRM = 1
DISCONFIRM = 2

class BeliefAgent(RnnAgent):

    @staticmethod
    def add_args(parser):
        super().add_args(parser)
        parser.add_argument(
            "--belief",
            choices = [
                "or", "and", "andor", "orand", "orandor",
            ],
            default = "or",
            help = "Partner response model",
        )

    # same init as RnnAgent, initialize belief in feed_context
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        response_pretrained_path = "../../response_annotation/save_pretrained"
        self.tokenizer = AutoTokenizer.from_pretrained(response_pretrained_path)
        self.confirmation_predictor = AutoModelForSequenceClassification.from_pretrained(
            response_pretrained_path)

    def feed_context(self, context, belief_constructor):
        super().feed_context(context, belief_constructor)
        self.num_dots = 7
        if self.args.belief == "or":
            self.belief = OrBelief(self.num_dots, context)
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
        self.next_mention_plans = []

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
            self.prior = self.belief.posterior(self.prior, side_info, 1)

        # add None to next_mention_plans, since reading turn => no generation
        self.next_mention_plans.append(None)


    def write(self, *args, **kwargs):
        # if write is given forced_words, then throw away plan. otherwise
        if kwargs["force_words"] is None:
            # generate next mention plan
            EdHs = self.belief.compute_EdHs(self.prior)
            cs, hs = self.belief.viz_belief(EdHs, n=4)
            plan = cs[0]
            # convert plan to dots_mentioned
            dots_mentioned = expand_plan(plan)
            # heuristic: produce next mentions per ref
            #dots_mentioned_per_ref_to_force = [dots_mentioned]
            # dots_mentioned_per_ref_to_force: num_mentions x bsz=1 x num_dots=7
            dots_mentioned_per_ref_to_force = dots_mentioned

            # clobber kwargs
            #kwargs["dots_mentioned_per_ref_to_force"] = dots_mentioned_per_ref_to_force
            #kwargs["force_dots_mentioned"] = True
            # if you force dots_mentioned, dont need to set num_markables
            # num_markables_to_force is just for forcing num_markables but letting model
            # generate mentions.
            #kwargs["dots_mentioned_num_markables_to_force"] = dots_mentioned_num_markables_to_force

            # override if not set
            kwargs.setdefault(
                "dots_mentioned_per_ref_to_force", dots_mentioned_per_ref_to_force)
            kwargs.setdefault("force_dots_mentioned", True)
        
        output = super().write(*args, **kwargs)

        if kwargs["force_words"] is not None:
            # set plan to the resolved refs of the forced utt
            plan = self.ref_preds[-1].any(0)[0].cpu().int().numpy()

        # add plan to next_mention_plan history
        self.next_mention_plans.append(plan)

        return output
