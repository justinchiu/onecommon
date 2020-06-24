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

def pairwise_differences(ctx, num_ent, dim_ent,
                         relation_include=['i_position', 'i_appearance', 'j_position', 'j_appearance'],
                         relation_include_angle=False,
                         symmetric=False
                         ):
    ents = ctx.view(ctx.size(0), num_ent, dim_ent)

    position, appearance = torch.split(ents, (2,2), dim=-1)

    rel_pairs = []

    for i in range(num_ent):
        # rel_pairs = []
        for j in range(num_ent):
            if i == j:
                continue
            if symmetric and i > j:
                continue
            dist = torch.sqrt((ents[:,i,0] - ents[:,j,0])**2 + (ents[:,i,1] - ents[:,j,1])**2)

            to_cat = []
            if 'i_position' in relation_include:
                to_cat.append(position[:,i])
            if 'i_appearance' in relation_include:
                to_cat.append(appearance[:,i])
            if 'j_position' in relation_include:
                to_cat.append(position[:,j])
            if 'j_appearance' in relation_include:
                to_cat.append(appearance[:,j])
            property_diff = ents[:,i,:] - ents[:,j,:]
            if symmetric:
                property_diff = torch.abs(property_diff)
            to_cat.extend([property_diff, dist.unsqueeze(1)])

            if relation_include_angle:
                if symmetric:
                    raise NotImplementedError("symmetric relation encoding that includes angles is not implemented")
                diff = position[:,i] - position[:,j]
                diff_x, diff_y = torch.split(diff, (1,1), dim=-1)
                rad_1 = torch.atan2(diff_x, diff_y) / math.pi
                rad_2 = torch.atan2(diff_y, diff_x) / math.pi
                to_cat.append(rad_1)
                to_cat.append(rad_2)

            rel_pairs.append((torch.cat(to_cat, 1).unsqueeze(1)))
        # ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
    ent_rel_pairs_t = torch.cat(rel_pairs, 1)
    if not symmetric:
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

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

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

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

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

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

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

        init_cont([self.relation_encoder, self.fc1], args.init_range)

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

        init_cont([self.fc1, self.fc2], args.init_range)

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

    def __init__(self, domain, args):
        super(RelationalAttentionContextEncoder3, self).__init__()
        self.args = args

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.relation_pooling = args.relation_pooling

        property_input_dim = 2 * len(self.args.properties_include)

        relation_input_dim = 2 * len(self.args.relation_include) + domain.dim_ent() + 1

        if property_input_dim == 0:
            property_output_dim = args.nembed_ctx
            relation_output_dim = args.nembed_ctx
            self.property_encoder = None
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

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

    def forward(self, ctx):
        relation_include_angle = hasattr(self, 'args') and self.args.relation_include_angle
        position, appearance, ent_rel_pairs = pairwise_differences(
            ctx, self.num_ent, self.dim_ent, self.args.relation_include, relation_include_angle, symmetric=False
        )
        rel_emb = self.relation_encoder(ent_rel_pairs)

        if self.relation_pooling == 'mean':
            rel_emb_pooled = rel_emb.mean(2)
        else:
            assert self.relation_pooling == 'max'
            rel_emb_pooled, _ = rel_emb.max(2)

        if self.property_encoder is not None:
            to_cat = []
            if 'position' in self.args.properties_include:
                to_cat.append(position)
            if 'appearance' in self.args.properties_include:
                to_cat.append(appearance)
            if len(to_cat) > 1:
                to_embed = torch.cat(to_cat, dim=-1)
            else:
                to_embed = to_cat[0]
            prop_emb = self.property_encoder(to_embed)
            out = torch.cat([prop_emb, rel_emb_pooled], 2)
        else:
            out = rel_emb_pooled
        return out
