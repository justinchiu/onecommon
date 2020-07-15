from collections import namedtuple

import torch

import utils

BELIEF_TYPES = [
    'none',
    'selected', 'partners',
    'last_partner_mentioned',
    't-2_partner_mentioned',
    'cumulative_partner_mentioned',
    'this_partner_mentioned',
    'this_partner_mentioned_predicted',
    'this_partner_mentioned_noised',
    'last_partner_mentioned_predicted',
    'this_mentioned',
    'this_mentioned_predicted',
    'last_mentioned_predicted',
    'cumulative_mentioned',
    'last_mentioned',
    't-2_mentioned',
    'next_mentioned',
]

_BeliefConstructor = namedtuple('_BeliefConstructor', [
    'args',
    'bsz', 'num_dots', 'inpts', 'ref_tgts', 'partner_ref_tgts_our_view', 'real_ids', 'partner_real_ids', 'sel_tgt',
    'is_self',
    'partner_dots_mentioned_our_view',
    'dots_mentioned',
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
        # parser.add_argument('--ref_beliefs_patterns', nargs='*')
        parser.add_argument('--partner_ref_beliefs', choices=BELIEF_TYPES, nargs='*', default=[])
        # parser.add_argument('--partner_beliefs_patterns', nargs='*')

        parser.add_argument('--belief_noise_pos_to_neg_probability', type=float, default=0.0)
        parser.add_argument('--belief_noise_neg_to_pos_probability', type=float, default=0.0)

    def make_beliefs(self, belief_type, timestep, partner_ref_outs, ref_outs):
        assert belief_type in [
            'selection_beliefs', 'generation_beliefs', 'mention_beliefs', 'ref_beliefs', 'partner_ref_beliefs'
        ]

        beliefs_names = vars(self.args)[belief_type]

        if timestep >= 0:
            if belief_type in ['generation_beliefs', 'ref_beliefs', 'partner_ref_beliefs']:
                assert len(partner_ref_outs) == timestep
                assert len(ref_outs) == timestep
            else:
                assert belief_type in ['mention_beliefs', 'selection_beliefs']
                assert len(partner_ref_outs) == timestep + 1
                assert len(ref_outs) == timestep + 1

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
                    'partner_mentioned', 'partner_mentioned_predicted', 'partner_mentioned_noised'
                ]
                if belief_to_use in ['mentioned', 'mentioned_noised']:
                    mentions = self.dots_mentioned
                    if belief_to_use == 'mention_noised':
                        mentions = [noise_beliefs(m, self.args.belief_noise_pos_to_neg_probability,
                                                  self.args.belief_noise_neg_to_pos_probability)
                                    if m is not None else None
                                    for m in mentions]
                elif belief_to_use == 'mentioned_predicted':
                    # TODO: unpack ref_outs to deal with structured outputs
                    mentions = [out[0].sigmoid().max(0).values
                                if out is not None else None
                                for out in ref_outs]
                    if self.args.detach_beliefs:
                        mentions = [m.detach() if m is not None else None for m in mentions]
                elif belief_to_use in ['partner_mentioned', 'partner_mentioned_noised']:
                    mentions = self.partner_dots_mentioned_our_view
                    if belief_to_use == 'partner_mentioned_noised':
                        mentions = [noise_beliefs(m, self.args.belief_noise_pos_to_neg_probability,
                                                  self.args.belief_noise_neg_to_pos_probability)
                                    if m is not None else None
                                    for m in mentions]
                else:
                    assert belief_to_use == 'partner_mentioned_predicted'
                    # TODO: use full mention logits if available rather than marginalized logits
                    mentions = [out[0].sigmoid().max(0).values
                                if out is not None else None
                                for out in partner_ref_outs]
                    if self.args.detach_beliefs:
                        mentions = [m.detach() if m is not None else None for m in mentions]
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
