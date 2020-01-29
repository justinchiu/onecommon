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

class AttentionContextEncoder(nn.Module):
    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, domain, args):
        super(AttentionContextEncoder, self).__init__()

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.property_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent(), int(args.nembed_ctx / 2)),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent() + 1, int(args.nembed_ctx / 2)),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.dropout = nn.Dropout(args.dropout)

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent)
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
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent)
        # only embed property and shape
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
            nn.Tanh()
        )

        self.fc1 = nn.Linear(num_rel * args.nhid_rel, args.nembed_ctx) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(args.dropout)

        init_cont([self.relation_encoder, self.fc1], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent)

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
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(args.dropout)

        init_cont([self.fc1, self.fc2], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        out = self.fc1(ctx_t)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.dropout(out)
        return out

