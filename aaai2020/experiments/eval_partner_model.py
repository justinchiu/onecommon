import argparse
import json
from tqdm import tqdm

import numpy as np
import torch

#from agent import *
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger, HierarchicalDialog
import models
from models.rnn_reference_model import RnnReferenceModel

from engines.rnn_reference_engine import (
    make_dots_mentioned,
    make_dots_mentioned_multi,
    make_dots_mentioned_per_ref_multi,
)

# 128 dot combinations
ndots = 7
configs = np.unpackbits(np.arange(128, dtype=np.uint8)[:,None], axis=1)[:,1:].astype(bool)
configs = torch.tensor(configs)
configs = configs.cuda()
int_configs = {
    4: configs[configs.sum(1) == 4],
    5: configs[configs.sum(1) == 5],
    6: configs[configs.sum(1) == 6],
}

# args
seed = 1
domain = "one_common"
data = "data/onecommon"
fold_num = 9
unk_threshold = 20
max_instances_per_split = None
max_mentions_per_utterance = None
crosstalk_split = None
batch_size = 1
threshold = 0.5
# / args

utils.use_cuda(True)
utils.set_seed(seed)

model_file = f"expts/rel3_tsel_ref_dial_model_separate/jc-partner/indicator-confirm-mean/{fold_num}/{fold_num}_best.th"
model = utils.load_model(model_file, prefix_dir=None, map_location="cpu")
model_type = model.args.model_type
model.args.wandb = False
model.eval()
model.cuda()

model_ty = models.get_model_type(model_type)
corpus = model_ty.corpus_ty(
    domain, data,
    train = f"train_reference_{fold_num}.txt",
    valid = f"valid_reference_{fold_num}.txt",
    test = f"test_reference_{fold_num}.txt",
    freq_cutoff = unk_threshold,
    verbose = True,
    max_instances_per_split = max_instances_per_split,
    max_mentions_per_utterance = max_mentions_per_utterance,
    crosstalk_split = crosstalk_split,
)


engine = model_ty.engine_ty(model, model.args, verbose=True)


def get_stats(dataset, N=100):
    stats = []
    with torch.no_grad():
        for batch in tqdm(dataset[:N]):
            chat_id = batch.chat_ids[0]

            id_int = batch.id_intersection
            nint = id_int.sum().item()
            int_config = int_configs[nint]

            # construct batches to compute posterior
            partner_model_losses = []
            for config in int_config:
                losses = engine._forward(batch._replace(id_intersection = config[None]), 0, corpus, False)
                intersect_loss = losses.next_partner_ref_intersect_loss
                confirm_loss = losses.next_partner_confirm_loss
                partner_model_losses.append((intersect_loss.item(), confirm_loss.item()))
            nlls = np.array([a+b for a,b in partner_model_losses])
            uprobs = np.exp(-nlls)
            probs = uprobs / uprobs.sum()

            # marginal probability of dot inclusion
            # only considers NEXT partner mention.
            # so if partner says first highly informative utterance,
            # our belief will be extremely uncalibrated.
            int_np = id_int.cpu().numpy()
            int_config_np = int_config.cpu().numpy()
            marginal_prob_inclusion = (int_config_np * probs[:,None]).sum(0)
            inclusion = marginal_prob_inclusion > threshold

            # batch info to variables, all torch
            confirms = batch.next_partner_confirm
            disconfirms = batch.next_partner_disconfirm
            # next partner refs
            next_ref_intersect = batch.next_partner_ref_intersect_ref
            next_ref_complement = batch.next_partner_ref_complement_ref
            our_refs = batch.ref_tgt
            # collapsed dot mentions, one for each turn
            our_dots_mentioned_multi = make_dots_mentioned_multi(
                our_refs, model.args, 1, 7)
            # dot mentions per turn + per reference
            our_dots_mentioned_per_ref_multi = make_dots_mentioned_per_ref_multi(
                our_refs, model.args, 1, 7)
            # / torch

            our_dots_mentioned_np = torch.cat(our_dots_mentioned_multi).sum(0).cpu().numpy()
            their_dots_mentioned_np = torch.cat(
                (next_ref_intersect, next_ref_complement), dim=1,
            ).sum(1).sum(0).cpu().numpy()

            print(f"Chat id {chat_id}")
            print("true intersection")
            print(int_np[0])
            print("our dots mentioned")
            print(our_dots_mentioned_np)
            print("their dots mentioned")
            print(their_dots_mentioned_np)
            print("marginal inclusion probs")
            print(" ".join(f"{x:.2f}" for x in marginal_prob_inclusion.tolist()))

            stats.append({
                "marginal_prob_inclusion": marginal_prob_inclusion.tolist(),
                "dot_intersection": int_np[0].tolist(),
                "partner_model_losses": partner_model_losses,
                "nlls": nlls.tolist(),
                "our_dots_mentioned": our_dots_mentioned_np.tolist(),
                "their_dots_mentioned": their_dots_mentioned_np.tolist(),
            })

    return stats



dataset, stats = corpus.train_dataset(1)
model_stats = get_stats(dataset)
with open("analysis/train_partner_model_stats.json", "w") as f:
    json.dump(model_stats, f)
dataset, stats = corpus.valid_dataset(1)
model_stats = get_stats(dataset)
with open("analysis/valid_partner_model_stats.json", "w") as f:
    json.dump(model_stats, f)


