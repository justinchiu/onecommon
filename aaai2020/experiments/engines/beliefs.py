import utils
import torch
from collections import namedtuple

_BeliefConstructor = namedtuple('_BeliefConstructor', [
    'args',
    'bsz', 'num_dots', 'inpts', 'ref_tgts', 'partner_ref_tgts_our_view', 'real_ids', 'partner_real_ids', 'sel_tgt', 'is_self',
    'partner_dots_mentioned_our_view',
    'dots_mentioned',
])

class BeliefConstructor(_BeliefConstructor):
    def make_beliefs(self, belief_type, timestep, partner_ref_outs):
        assert belief_type in ['selection_beliefs', 'generation_beliefs', 'mention_beliefs']

        beliefs_names = vars(self.args)[belief_type]

        if timestep >= 0:
            if belief_type in ['generation_beliefs']:
                assert len(partner_ref_outs) == timestep
            else:
                assert belief_type in ['mention_beliefs', 'selection_beliefs']
                assert len(partner_ref_outs) == timestep + 1

        all_beliefs = []
        for beliefs_name in beliefs_names:
            if beliefs_name == 'selected':
                # bsz x num_dots x 1, one-hot if that dot is the one selected
                beliefs = torch.zeros(self.sel_tgt.size(0), self.num_dots).scatter(1, self.sel_tgt.unsqueeze(1), 1).unsqueeze(-1)
            elif beliefs_name == 'partners':
                partner_has = []
                for batch_ix, (a_rids, p_rids) in enumerate(utils.safe_zip(self.real_ids, self.partner_real_ids)):
                    p_rids = set(p_rids)
                    partner_has.append([a_rid in p_rids for a_rid in a_rids])
                beliefs = torch.tensor(partner_has).float().unsqueeze(-1)
            elif beliefs_name in ['last_partner_mentioned', 'last_partner_mentioned_predicted']:
                if beliefs_name == 'last_partner_mentioned':
                    mentions = self.partner_dots_mentioned_our_view
                else:
                    if partner_ref_outs and partner_ref_outs[0] is None:
                        raise Exception("must pass --partner_reference_prediction with --this_partner_mentioned_predicted")
                    mentions = [
                        # out.sigmoid().max(0).values if out is not None else None
                        # take the max probability over the dot mentions
                        # out_logits: num_mentions x bsz x num_dots
                        out_logits.sigmoid().max(0).values
                        for out_logits, _ in partner_ref_outs
                    ]
                    if self.args.detach_beliefs:
                        mentions = [
                            mention.detach() if mention is not None else None
                            for mention in mentions
                        ]
                    mentions = [
                        mention if mention is not None else torch.zeros(self.bsz, self.num_dots)
                        for mention in mentions
                    ]
                beliefs = torch.zeros(self.bsz, self.num_dots)
                if timestep > 0:
                    ts = (torch.tensor([timestep] * self.bsz).long() - 1) - self.is_self[timestep-1].long()
                    for j, t_j in enumerate(ts):
                        if t_j >= 0:
                            beliefs[j] = mentions[t_j][j]
                beliefs = beliefs.float().unsqueeze(-1)
            elif beliefs_name == 'this_partner_mentioned':
                if timestep >= 0:
                    beliefs = self.partner_dots_mentioned_our_view[timestep].float().unsqueeze(-1)
                else:
                    beliefs = torch.zeros_like(self.partner_dots_mentioned_our_view[0]).float().unsqueeze(-1)
            elif beliefs_name == 'this_partner_mentioned_predicted':
                if timestep >= 0:
                    if partner_ref_outs[timestep] is None:
                        raise Exception("must pass --partner_reference_prediction with --this_partner_mentioned_predicted")
                    ref_logits, ref_full = partner_ref_outs[timestep]
                    beliefs = ref_logits.sigmoid().max(0).values
                    beliefs = beliefs.unsqueeze(-1)
                    if self.args.detach_beliefs:
                        beliefs = beliefs.detach()
            elif beliefs_name == 'cumulative_partner_mentioned':
                beliefs = torch.zeros(self.bsz, self.num_dots).bool()
                if timestep >= 0:
                    for t in range(timestep):
                        beliefs |= self.partner_dots_mentioned_our_view[t]
                beliefs = beliefs.float().unsqueeze(-1)
            elif beliefs_name == 'this_mentioned':
                if timestep >= 0:
                    beliefs = self.dots_mentioned[timestep].float().unsqueeze(-1)
                else:
                    beliefs = torch.zeros_like(self.dots_mentioned[0]).float().unsqueeze(-1)
            elif beliefs_name == 'last_mentioned':
                beliefs = torch.zeros(self.bsz, self.num_dots)
                if timestep > 0:
                    ts = (torch.tensor([timestep] * self.bsz).long() - 1) - (1 - self.is_self[timestep-1].long())
                    for j, t_j in enumerate(ts):
                        if t_j >= 0:
                            beliefs[j] = self.dots_mentioned[t_j][j]
                beliefs = beliefs.float().unsqueeze(-1)
            elif beliefs_name == 'next_mentioned':
                if timestep < len(self.dots_mentioned) - 1:
                    beliefs = self.dots_mentioned[timestep+1].float().unsqueeze(-1)
                else:
                    beliefs = torch.zeros_like(self.dots_mentioned[0]).float().unsqueeze(-1)
            elif beliefs_name == 'cumulative_mentioned':
                beliefs = torch.zeros(self.bsz, self.num_dots).bool()
                if timestep >= 0:
                    for t in range(timestep):
                        beliefs |= self.dots_mentioned[t]
                beliefs = beliefs.float().unsqueeze(-1)
            else:
                raise ValueError('invalid --{} {}'.format(self.arg_name, beliefs_name))
            all_beliefs.append(beliefs)

        if all_beliefs:
            if len(all_beliefs) > 1:
                return torch.cat(all_beliefs, dim=-1)
            else:
                return all_beliefs[0]
        else:
            return None
    #
    # if hasattr(args, 'selection_beliefs'):
    #     if timestep == len(inpts) - 1:
    #         selection_beliefs = make_beliefs(args.selection_beliefs, 'selection_beliefs')
    #     else:
    #         selection_beliefs = None
    # else:
    #     selection_beliefs = None
    # if hasattr(args, 'generation_beliefs'):
    #     generation_beliefs = make_beliefs(args.generation_beliefs, 'generation_beliefs')
    # else:
    #     generation_beliefs = None
    #
    # if hasattr(args, 'mention_beliefs'):
    #     mention_beliefs = make_beliefs(args.mention_beliefs, 'mention_beliefs')
    # else:
    #     mention_beliefs = None
    #
    # return selection_beliefs, generation_beliefs, mention_beliefs
    #
