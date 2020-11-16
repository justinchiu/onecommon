"""
Set of context encoders.
"""
from itertools import combinations
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from models.utils import *

def single_difference(ent_i, ent_j, relation_include, relation_include_angle, include_symmetric_rep, include_asymmetric_rep):
    # ent_i: ... x 4
    # ent_j: ... x 4
    assert ent_i.size(-1) == 4
    assert ent_j.size(-1) == 4
    dist = torch.sqrt((ent_i[...,0] - ent_j[...,0])**2 + (ent_i[...,1] - ent_j[...,1])**2)

    position_i, appearance_i = torch.split(ent_i, (2,2), dim=-1)
    position_j, appearance_j = torch.split(ent_j, (2,2), dim=-1)

    to_cat = []
    if 'i_position' in relation_include:
        to_cat.append(position_i)
    if 'i_appearance' in relation_include:
        to_cat.append(appearance_i)
    if 'j_position' in relation_include:
        to_cat.append(position_j)
    if 'j_appearance' in relation_include:
        to_cat.append(appearance_j)
    property_diff = ent_i - ent_j

    if include_symmetric_rep:
        to_cat.append(torch.abs(property_diff))
    if include_asymmetric_rep:
        to_cat.append(property_diff)
    to_cat.append(dist.unsqueeze(-1))

    if relation_include_angle:
        if not include_asymmetric_rep:
            raise NotImplementedError("a symmetric representation that includes angles isn't implemented")
        diff = position_i - position_j
        diff_x, diff_y = torch.split(diff, (1,1), dim=-1)
        rad_1 = torch.atan2(diff_x, diff_y) / math.pi
        rad_2 = torch.atan2(diff_y, diff_x) / math.pi
        to_cat.append(rad_1)
        to_cat.append(rad_2)
    return torch.cat(to_cat, -1)

def pairwise_differences(ctx, num_ent, dim_ent,
                         relation_include=['i_position', 'i_appearance', 'j_position', 'j_appearance'],
                         relation_include_angle=False,
                         symmetric=False,
                         include_asymmetric_rep_in_symmetric=False,
                         include_self=False,
                         ):
    ents = ctx.view(ctx.size(0), num_ent, dim_ent)

    position, appearance = torch.split(ents, (2,2), dim=-1)

    rel_pairs = []

    for i in range(num_ent):
        # rel_pairs = []
        for j in range(num_ent):
            if (not include_self) and i == j:
                continue
            if symmetric and i > j:
                continue
            diff = single_difference(ents[:,i], ents[:,j], relation_include, relation_include_angle,
                                     symmetric, (not symmetric) or include_asymmetric_rep_in_symmetric)
            rel_pairs.append(diff.unsqueeze(1))
        # ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
    ent_rel_pairs_t = torch.cat(rel_pairs, 1)
    if not symmetric:
        if include_self:
            assert len(rel_pairs) == num_ent * num_ent
            ent_rel_pairs_t = ent_rel_pairs_t.view(ent_rel_pairs_t.size(0), num_ent, num_ent, ent_rel_pairs_t.size(-1))
        else:
            assert len(rel_pairs) == num_ent * (num_ent - 1)
            ent_rel_pairs_t = ent_rel_pairs_t.view(ent_rel_pairs_t.size(0), num_ent, num_ent-1, ent_rel_pairs_t.size(-1))
    else:
        assert len(rel_pairs) == num_ent * (num_ent - 1) // 2
    return position, appearance, ent_rel_pairs_t

class AttentionContextEncoder(nn.Module):
    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, domain, args):
        super(AttentionContextEncoder, self).__init__()
        self.args = args

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.property_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent(), int(args.nembed_ctx / 2)),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent() + 1, int(args.nembed_ctx / 2)),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, ctx):
        ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)
        prop_emb = self.property_encoder(ents)
        ent_rel_pairs = []
        for i in range(self.num_ent):
            rel_pairs = []
            for j in range(self.num_ent):
                if i == j:
                    continue
                dist = torch.sqrt((ents[:,i,0] - ents[:,j,0])**2 + (ents[:,i,1] - ents[:,j,1])**2)
                rel_pairs.append((torch.cat([ents[:,i,:] - ents[:,j,:], dist.unsqueeze(1)], 1).unsqueeze(1)))
            ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
        ent_rel_pairs = torch.cat(ent_rel_pairs, 1)
        rel_emb = self.relation_encoder(ent_rel_pairs).sum(2)
        out = torch.cat([prop_emb, rel_emb], 2)
        return out

class RelationalAttentionContextEncoder(nn.Module):
    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, domain, args):
        super(RelationalAttentionContextEncoder, self).__init__()
        self.args = args

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.property_encoder = nn.Sequential(
            torch.nn.Linear(2, int(args.nembed_ctx / 2)),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(2 + domain.dim_ent() + 1, int(args.nembed_ctx / 2)),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, ctx):
        ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)
        # only embed color and size
        prop_emb = self.property_encoder(ents[:,:,2:])
        ent_rel_pairs = []
        for i in range(self.num_ent):
            rel_pairs = []
            for j in range(self.num_ent):
                if i == j:
                    continue
                dist = torch.sqrt((ents[:,i,0] - ents[:,j,0])**2 + (ents[:,i,1] - ents[:,j,1])**2)
                rel_pairs.append((torch.cat([ents[:,i,:2], ents[:,i,:] - ents[:,j,:], dist.unsqueeze(1)], 1).unsqueeze(1)))
            ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
        ent_rel_pairs = torch.cat(ent_rel_pairs, 1)
        rel_emb = self.relation_encoder(ent_rel_pairs).sum(2)
        out = torch.cat([prop_emb, rel_emb], 2)
        return out

class RelationalAttentionContextEncoder2(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--relation_encoder_layers', type=int, choices=[1,2], default=1)
        parser.add_argument('--relation_pooling', choices=['mean', 'max'], default='mean')
        parser.add_argument('--relation_ablate_properties', action='store_true')
        parser.add_argument('--relation_include_angle', action='store_true') # doesn't seem to help
        parser.add_argument('--property_include_coordinates', action='store_true') # doesn't seem to help

    def __init__(self, domain, args):
        super(RelationalAttentionContextEncoder2, self).__init__()
        self.args = args

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.relation_pooling = args.relation_pooling

        # default: only embed color and size
        property_input_dim = 2
        if args.property_include_coordinates:
            property_input_dim += 2

        # default: color and size for both objects, and subtracted representation, and distance
        relation_input_dim = 4 + domain.dim_ent() + 1
        if args.relation_include_angle:
            relation_input_dim += 2

        if args.relation_ablate_properties:
            self.property_encoder = None
            property_output_dim = args.nembed_ctx
            relation_output_dim = args.nembed_ctx
        else:
            property_output_dim = int(args.nembed_ctx / 2)
            relation_output_dim = int(args.nembed_ctx / 2)
            self.property_encoder = nn.Sequential(
                torch.nn.Linear(property_input_dim, property_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )

        if args.relation_encoder_layers == 2:
            hidden_dim = relation_output_dim
            self.relation_encoder = nn.Sequential(
                torch.nn.Linear(relation_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(hidden_dim, relation_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
        else:
            assert args.relation_encoder_layers == 1
            self.relation_encoder = nn.Sequential(
                torch.nn.Linear(relation_input_dim, relation_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )

    def forward(self, ctx):
        ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)
        ent_rel_pairs = []
        for i in range(self.num_ent):
            rel_pairs = []
            for j in range(self.num_ent):
                if i == j:
                    continue
                dist = torch.sqrt((ents[:,i,0] - ents[:,j,0])**2 + (ents[:,i,1] - ents[:,j,1])**2)
                to_cat = [ents[:,i,2:], ents[:,j,2:], ents[:,i,:] - ents[:,j,:], dist.unsqueeze(1)]
                if hasattr(self, 'args') and self.args.relation_include_angle:
                    diff = ents[:,i,:2] - ents[:,j,:2]
                    diff_x, diff_y = torch.split(diff, (1,1), dim=-1)
                    rad_1 = torch.atan2(diff_x, diff_y) / math.pi
                    rad_2 = torch.atan2(diff_y, diff_x) / math.pi
                    to_cat.append(rad_1)
                    to_cat.append(rad_2)
                rel_pairs.append((torch.cat(to_cat, 1).unsqueeze(1)))
            ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
        ent_rel_pairs = torch.cat(ent_rel_pairs, 1)
        rel_emb = self.relation_encoder(ent_rel_pairs)
        if hasattr(self, 'relation_pooling'):
            if self.relation_pooling == 'mean':
                rel_emb_pooled = rel_emb.mean(2)
            else:
                assert self.relation_pooling == 'max'
                rel_emb_pooled, _ = rel_emb.max(2)
        else:
            rel_emb_pooled, _ = rel_emb.max(2)
        if self.property_encoder is not None:
            if hasattr(self, 'args') and self.args.property_include_coordinates:
                prop_emb = self.property_encoder(ents)
            else:
                prop_emb = self.property_encoder(ents[:,:,2:])
            out = torch.cat([prop_emb, rel_emb_pooled], 2)
        else:
            out = rel_emb_pooled
        return out

class RelationContextEncoder(nn.Module):
    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, domain, args):
        super(RelationContextEncoder, self).__init__()

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()
        num_rel = int(domain.num_ent() * (domain.num_ent() - 1) / 2)

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(2 * domain.dim_ent(), args.nhid_rel),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(num_rel * args.nhid_rel, args.nembed_ctx) 
        self.tanh = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, ctx):
        ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)

        rel_pairs = []
        for i in range(self.num_ent):
            for j in range(self.num_ent):
                if i < j:
                    rel_pairs.append(torch.cat([ents[:,i,:],ents[:,j,:]], 1).unsqueeze(1))
        rel_pairs = torch.cat(rel_pairs, 1)        
        out = self.relation_encoder(rel_pairs).view(rel_pairs.size(0), -1)
        out = self.fc1(out)
        out = self.tanh(out)
        out = self.dropout(out)
        return out


class MlpContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""

    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, domain, args):
        super(MlpContextEncoder, self).__init__()

        self.fc1 = nn.Linear(domain.num_ent() * domain.dim_ent(), args.nembed_ctx)
        self.fc2 = nn.Linear(args.nembed_ctx, args.nembed_ctx)
        self.tanh = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, ctx):
        out = self.fc1(ctx)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.dropout(out)
        return out

class RelationalAttentionContextEncoder3(nn.Module):
    @classmethod
    def add_args(cls, parser):
        # parser.add_argument('--relation_encoder_layers', type=int, choices=[1,2], default=1)
        # parser.add_argument('--relation_pooling', choices=['mean', 'max'], default='mean')
        parser.add_argument('--properties_include',
                            choices=['position', 'appearance'],
                            default=['appearance'],
                            nargs='*')
        parser.add_argument('--relation_include',
                            choices=['i_position', 'i_appearance', 'j_position', 'j_appearance'],
                            default=['i_appearance', 'j_appearance'],
                            nargs='*')
        parser.add_argument('--encode_relative_to_extremes', action='store_true')

    def __init__(self, domain, args):
        super(RelationalAttentionContextEncoder3, self).__init__()
        self.args = args

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.relation_pooling = args.relation_pooling

        property_input_dim = 2 * len(self.args.properties_include)

        relation_input_dim = 2 * len(self.args.relation_include) + domain.dim_ent() + 1

        if args.encode_relative_to_extremes:
            assert args.nembed_ctx % 4 == 0
            extremes_output_dim = args.nembed_ctx // 4
            remaining_dims = args.nembed_ctx - extremes_output_dim
        else:
            extremes_output_dim = 0
            remaining_dims = args.nembed_ctx

        if property_input_dim == 0:
            property_output_dim = 0
            relation_output_dim = remaining_dims
            self.property_encoder = None
        else:
            property_output_dim = remaining_dims // 2
            relation_output_dim = remaining_dims - property_output_dim
            self.property_encoder = nn.Sequential(
                torch.nn.Linear(property_input_dim, property_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )

        assert property_output_dim + relation_output_dim + extremes_output_dim == args.nembed_ctx

        if args.encode_relative_to_extremes:
            self.extremes_encoder = nn.Sequential(
                torch.nn.Linear(relation_input_dim*2, extremes_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
        else:
            self.extremes_encoder = None

        if args.relation_encoder_layers == 2:
            hidden_dim = relation_output_dim
            self.relation_encoder = nn.Sequential(
                torch.nn.Linear(relation_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(hidden_dim, relation_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
        else:
            assert args.relation_encoder_layers == 1
            self.relation_encoder = nn.Sequential(
                torch.nn.Linear(relation_input_dim, relation_output_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )

    def forward(self, ctx, relational_dot_mask=None):
        relation_include_angle = hasattr(self, 'args') and self.args.relation_include_angle
        bsz = ctx.size(0)
        position, appearance, ent_rel_pairs = pairwise_differences(
            ctx, self.num_ent, self.dim_ent, self.args.relation_include, relation_include_angle,
            symmetric=False, include_self=(relational_dot_mask is not None)
        )
        rel_emb = self.relation_encoder(ent_rel_pairs)

        if self.relation_pooling == 'mean':
            if relational_dot_mask is not None:
                # rel_emb: bsz x num_dots x num_dots x hidden_dim
                assert relational_dot_mask.size() == (bsz, self.num_ent)
                dot_mask_select = torch.einsum("bx,by->bxy", (relational_dot_mask,relational_dot_mask))
                # remove diagonals
                # remove diagonal entries
                dot_mask_select *= (1-torch.eye(self.num_ent)).unsqueeze(0).expand_as(dot_mask_select)
                # rel_emb_pooled: bsz x num_dots x hidden_dim
                rel_emb_pooled = (rel_emb * dot_mask_select.unsqueeze(-1).expand_as(rel_emb)).sum(2)
                rel_emb_pooled = rel_emb_pooled / torch.clamp_min(relational_dot_mask.sum(1) - 1, 1.0).unsqueeze(-1).unsqueeze(-1)
            else:
                rel_emb_pooled = rel_emb.mean(2)
        else:
            assert self.relation_pooling == 'max'
            if relational_dot_mask is not None:
                raise NotImplementedError()
            rel_emb_pooled, _ = rel_emb.max(2)

        to_cat = []

        if self.property_encoder is not None:
            prop_to_cat = []
            if 'position' in self.args.properties_include:
                prop_to_cat.append(position)
            if 'appearance' in self.args.properties_include:
                prop_to_cat.append(appearance)
            if len(prop_to_cat) > 1:
                to_embed = torch.cat(prop_to_cat, dim=-1)
            else:
                to_embed = prop_to_cat[0]
            prop_emb = self.property_encoder(to_embed)
            to_cat.append(prop_emb)

        to_cat.append(rel_emb_pooled)

        if vars(self.args).get('encode_relative_to_extremes'):
            ents = ctx.view(ctx.size(0), self.num_ent, self.dim_ent)
            # bsz x num_ent x relation_input_dim
            ex_min = ents.min(1).values.unsqueeze(1).expand_as(ents)
            # bsz x num_ent x relation_input_dim
            ex_max = ents.max(1).values.unsqueeze(1).expand_as(ents)
            # bsz x num_ent x relation_input_dim
            diffs_min = single_difference(ents, ex_min, self.args.relation_include, relation_include_angle,
                                          include_symmetric_rep=False, include_asymmetric_rep=True)
            diffs_max = single_difference(ents, ex_max, self.args.relation_include, relation_include_angle,
                                          include_symmetric_rep=False, include_asymmetric_rep=True)
            diffs = torch.cat((diffs_min, diffs_max), dim=-1)
            extremes_emb = self.extremes_encoder(diffs)
            to_cat.append(extremes_emb)

        if len(to_cat) > 1:
            out = torch.cat(to_cat, 2)
        else:
            out = to_cat[0]
        return out
