from collections import namedtuple

import torch

import utils

BELIEF_TYPES = [
    'none',
    'selected', 'partners',
    'next_mentioned',

    # this mentions
    'this_mentioned', 'this_mentioned_predicted',
    'last_mentioned', 'last_mentioned_predicted',
    'last_primary_mentioned', 'last_primary_mentioned_predicted',
    't-2_mentioned',
    'cumulative_mentioned',

    # partner mentions
    'this_partner_mentioned', 'this_partner_mentioned_predicted', 'this_partner_mentioned_noised',
    'this_partner_primary_mentioned', 'this_partner_primary_mentioned_predicted',
    'last_partner_mentioned', 'last_partner_mentioned_predicted',
    'last_partner_primary_mentioned', 'last_partner_primary_mentioned_predicted',
    't-2_partner_mentioned',
    'cumulative_partner_mentioned',

    # either mentioned
    'this_either_mentioned', 'this_either_mentioned_predicted',
    'last_either_mentioned', 'last_either_mentioned_predicted',
]

_BeliefConstructor = namedtuple('_BeliefConstructor', [
    'args', 'epoch',
    'bsz', 'num_dots', 'inpts', 'ref_tgts', 'partner_ref_tgts_our_view', 'real_ids', 'partner_real_ids', 'sel_tgt',
    'is_self',
    'partner_dots_mentioned_our_view',
    'partner_dots_mentioned_our_view_per_ref',
    'dots_mentioned',
    'dots_mentioned_per_ref',
    'ref_inpts',
    'partner_ref_inpts',
    'num_markables',
    'partner_num_markables',
])

def noise_beliefs(zero_one_tensor, pos_to_neg_prob, neg_to_pos_prob, num_samples=1):
    # for every entry with a 0, have a neg_to_pos_prob chance of drawing a 1; for every entry with a 1, have a pos_to_neg_prob chance of drawing a zero
    draw_probs = torch.where(zero_one_tensor == 1.0,
                             torch.tensor(1 - pos_to_neg_prob),
                             torch.tensor(neg_to_pos_prob))
    dist = torch.distributions.Bernoulli(draw_probs)
    if num_samples > 1:
        return dist.sample((num_samples,))
    else:
        return dist.sample()


class BeliefConstructor(_BeliefConstructor):
    @staticmethod
    def add_belief_args(parser):
        parser.add_argument('--selection_beliefs', choices=BELIEF_TYPES, nargs='*',
                            default=[],
                            help='selected: indicator on what you chose. partners: indicator on what the other person has')
        # parser.add_argument('--selection_beliefs_patterns', nargs='*')

        parser.add_argument('--generation_beliefs', choices=BELIEF_TYPES, nargs='*',
                            default=[],
                            help='selected: indicator on what you chose. partners: indicator on what the other person has')
        # parser.add_argument('--generation_beliefs_patterns', nargs='*')

        parser.add_argument('--mention_beliefs', choices=BELIEF_TYPES, nargs='*', default=[])
        # parser.add_argument('--mention_beliefs_patterns', nargs='*')
        parser.add_argument('--ref_beliefs', choices=BELIEF_TYPES, nargs='*', default=[])
        parser.add_argument('--ref_beliefs_warmup', choices=BELIEF_TYPES, nargs='*', default=[])
        parser.add_argument('--ref_beliefs_warmup_epochs', type=int)
        # parser.add_argument('--ref_beliefs_patterns', nargs='*')
        parser.add_argument('--partner_ref_beliefs', choices=BELIEF_TYPES, nargs='*', default=[])
        parser.add_argument('--partner_ref_beliefs_warmup', choices=BELIEF_TYPES, nargs='*', default=[])
        parser.add_argument('--partner_ref_beliefs_warmup_epochs', type=int)
        # parser.add_argument('--partner_beliefs_patterns', nargs='*')

        parser.add_argument('--belief_noise_pos_to_neg_probability', type=float, default=0.0)
        parser.add_argument('--belief_noise_neg_to_pos_probability', type=float, default=0.0)

    def make_beliefs(self, belief_type, timestep, partner_ref_outs, ref_outs):
        if belief_type == 'next_mention_latents':
            # TODO: maybe implement this; currently it's only used in the dot_recurrence
            return None
        assert belief_type in [
            'selection', 'generation', 'mention', 'ref', 'partner_ref',
        ]

        if belief_type in ['ref', 'partner_ref'] and self.epoch is not None:
            if belief_type == 'ref':
                if self.args.ref_beliefs_warmup_epochs is not None and self.epoch <= self.args.ref_beliefs_warmup_epochs:
                    beliefs_names = self.args.ref_beliefs_warmup
                else:
                    beliefs_names = self.args.ref_beliefs
            if belief_type == 'partner_ref':
                if self.args.partner_ref_beliefs_warmup_epochs is not None and self.epoch <= self.args.partner_ref_beliefs_warmup_epochs:
                    beliefs_names = self.args.partner_ref_beliefs_warmup
                else:
                    beliefs_names = self.args.partner_ref_beliefs
        else:
            beliefs_names = vars(self.args)[f'{belief_type}_beliefs']

        if timestep >= 0:
            if belief_type in ['generation', 'ref', 'partner_ref']:
                assert len(partner_ref_outs) == timestep
                assert len(ref_outs) == timestep
            else:
                assert belief_type in ['mention', 'selection']
                assert len(partner_ref_outs) == timestep + 1
                assert len(ref_outs) == timestep + 1

        def get_self_predicted(primary):
            # import here to avoid a circular import
            from models.reference_predictor import ReferencePredictor
            reference_predictor = ReferencePredictor(self.args)
            preds = [
                reference_predictor.forward(ref_inpt, ref_tgt, ref_out, this_num_markables)[1]
                for ref_inpt, ref_tgt, ref_out, this_num_markables in
                zip(self.ref_inpts, self.ref_tgts, ref_outs, self.num_markables)
            ]
            if not primary:
                mentions = [
                    pred.max(0).values if pred is not None else None
                    for pred in preds
                ]
            else:
                mentions = [
                    pred[0] if pred is not None else None
                    for pred in preds
                ]
            # TODO: is this still necessary?
            if self.args.detach_beliefs:
                mentions = [m.detach() if m is not None else None for m in mentions]
            return mentions

        def get_partner_predicted(primary):
            # import here to avoid a circular import
            from models.reference_predictor import ReferencePredictor
            reference_predictor = ReferencePredictor(self.args)
            preds = [
                reference_predictor.forward(partner_ref_inpt, partner_ref_tgt, partner_ref_out, this_partner_num_markables)[1]
                for partner_ref_inpt, partner_ref_tgt, partner_ref_out, this_partner_num_markables in
                zip(self.partner_ref_inpts, self.partner_ref_tgts_our_view, partner_ref_outs, self.partner_num_markables)
            ]
            if not primary:
                mentions = [
                    pred.max(0).values if pred is not None else None
                    for pred in preds
                ]
            else:
                mentions = [
                    pred[0] if pred is not None else None
                    for pred in preds
                ]
            if self.args.detach_beliefs:
                mentions = [m.detach() if m is not None else None for m in mentions]
            return mentions

        def combine_either(self_mentions, partner_mentions):
            mentions = []
            for this_is_self, this_self_mentions, this_partner_mentions in zip(
                self.is_self, self_mentions, partner_mentions
            ):
                if this_partner_mentions is None:
                    mentions.append(this_self_mentions)
                elif this_self_mentions is None:
                    mentions.append(this_partner_mentions)
                else:
                    assert this_self_mentions.size() == this_partner_mentions.size()
                    this_mentions = this_partner_mentions.clone()
                    this_mentions[this_is_self] = this_self_mentions[this_is_self]
                    mentions.append(this_mentions)
            return mentions

        all_beliefs = []
        for beliefs_name in beliefs_names:
            if beliefs_name == 'selected':
                # bsz x num_dots x 1, one-hot if that dot is the one selected
                beliefs = torch.zeros(self.sel_tgt.size(0), self.num_dots).scatter(1, self.sel_tgt.unsqueeze(1),
                                                                                   1)
            elif beliefs_name == 'partners':
                partner_has = []
                for batch_ix, (a_rids, p_rids) in enumerate(utils.safe_zip(self.real_ids, self.partner_real_ids)):
                    p_rids = set(p_rids)
                    partner_has.append([a_rid in p_rids for a_rid in a_rids])
                beliefs = torch.tensor(partner_has)
            else:
                tokens = beliefs_name.split('_')
                timestep_to_use_name = tokens[0] # e.g.
                assert timestep_to_use_name in ['next', 'this', 'last', 't-2', 'cumulative']
                belief_to_use = '_'.join(tokens[1:])
                assert belief_to_use in [
                    'mentioned', 'mentioned_predicted', 'mentioned_noised',
                    'partner_mentioned', 'partner_mentioned_predicted', 'partner_mentioned_noised',
                    'primary_mentioned', 'primary_mentioned_predicted',
                    'partner_primary_mentioned', 'partner_primary_mentioned_predicted',
                    'either_mentioned', 'either_mentioned_predicted',
                ]
                if belief_to_use in ['mentioned', 'mentioned_noised']:
                    mentions = self.dots_mentioned
                    if belief_to_use == 'mention_noised':
                        mentions = [noise_beliefs(m, self.args.belief_noise_pos_to_neg_probability,
                                                  self.args.belief_noise_neg_to_pos_probability)
                                    if m is not None else None
                                    for m in mentions]
                elif belief_to_use == 'primary_mentioned':
                    # use the dots mentioned in the first mention per utterance
                    # self.dots_mentioned_per_ref is a list with one tensor for each utterance, of size
                    # bsz x num_mentions x num_dots
                    mentions = [utt_mentions[:,0] if (utt_mentions is not None and utt_mentions.size(1) > 0) else None
                                for utt_mentions in self.dots_mentioned_per_ref]
                elif belief_to_use == 'partner_primary_mentioned':
                    # use the dots mentioned in the first mention per utterance
                    mentions = [utt_mentions[:,0] if (utt_mentions is not None and utt_mentions.size(1) > 0) else None
                                for utt_mentions in self.partner_dots_mentioned_our_view_per_ref]
                elif belief_to_use in ['mentioned_predicted', 'primary_mentioned_predicted']:
                    mentions = get_self_predicted(primary=belief_to_use == 'primary_mentioned_predicted')
                elif belief_to_use in ['partner_mentioned', 'partner_mentioned_noised']:
                    mentions = self.partner_dots_mentioned_our_view
                    if belief_to_use == 'partner_mentioned_noised':
                        mentions = [noise_beliefs(m, self.args.belief_noise_pos_to_neg_probability,
                                                  self.args.belief_noise_neg_to_pos_probability)
                                    if m is not None else None
                                    for m in mentions]
                elif belief_to_use in ['partner_mentioned_predicted', 'partner_primary_mentioned_predicted']:
                    mentions = get_partner_predicted(primary=belief_to_use == 'partner_primary_mentioned_predicted')
                elif belief_to_use == 'either_mentioned':
                    mentions = combine_either(
                        self.dots_mentioned, self.partner_dots_mentioned_our_view
                    )
                    pass
                elif belief_to_use == 'either_mentioned_predicted':
                    mentions = combine_either(
                        get_self_predicted(primary=False), get_partner_predicted(primary=False),
                    )
                else:
                    raise ValueError(f"invalid belief_to_use {belief_to_use}")
                if timestep_to_use_name == 'cumulative':
                    beliefs = torch.zeros(self.bsz, self.num_dots).bool()
                    if timestep >= 0:
                        for t in range(timestep):
                            if mentions[t] is not None:
                                beliefs |= mentions[t]
                elif timestep_to_use_name == 'this':
                    if timestep >= 0 and mentions[timestep] is not None:
                        beliefs = mentions[timestep]
                    else:
                        beliefs = torch.zeros(self.bsz, self.num_dots)
                else:
                    if timestep_to_use_name == 'next':
                        direction = +1
                        offset = 1
                    elif timestep_to_use_name == 'last':
                        direction = -1
                        offset = 1
                    elif timestep_to_use_name == 't-2':
                        direction = -1
                        offset = 2

                    timestep_to_use = timestep+direction*offset
                    if timestep_to_use < 0 or timestep_to_use >= len(self.is_self):
                        beliefs = torch.zeros(self.bsz, self.num_dots)
                    else:
                        ts = torch.full((self.bsz,), timestep_to_use).long()
                        is_self = self.is_self[timestep_to_use]
                        shift = is_self if belief_to_use.startswith('partner') else (~is_self)

                        ts = ts + direction*shift.long()

                        beliefs = torch.zeros(self.bsz, self.num_dots)
                        for ix, t in enumerate(ts):
                            if 0 <= t < len(mentions) and mentions[t] is not None:
                                beliefs[ix] = mentions[t][ix]

            all_beliefs.append(beliefs.float().unsqueeze(-1))

        if all_beliefs:
            if len(all_beliefs) > 1:
                return torch.cat(all_beliefs, dim=-1)
            else:
                return all_beliefs[0]
        else:
            return None

class BlankBeliefConstructor(object):
    def make_beliefs(self, *args):
        return None