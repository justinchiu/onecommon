from typing import NamedTuple
from collections import defaultdict

from pathlib import Path
import numpy as np
from jinja2 import Template
from rich.progress import track
import json
import itertools
import math

import torch

from belief import process_ctx

from datasets import Dataset, load_dataset

from domain import get_domain

from corpora.data import Dictionary, get_tag
from corpora.reference_sentence import ReferenceSentenceCorpus

import template

from belief_utils import is_contiguous

import bitutils
from hfutils import (
    HfDataOptions, Property, DescriptionFormat,
    construct_feature_string, get_bart_tokenizer,
    GenerationExtras,
)

fields = (
    "chat_id",
    "scenario_id",
    "agent",
    "raw_dots",
    "dots",
    "plan_specific_dots",
    "mention_specific_dots",
    "text", # all prev turns
    "plan",
    "mentions",
    "raw_mentions",
    "joint_mentions",
    "markers",
    "amarkers",
    "tags",
    "partner_mentions", # for ref res
    "raw_partner_mentions", # for ref res
    "joint_partner_mentions", # for ref res
    "confirmation",
    "selection_leaning",
    "selection",
    "coref",
    "outtext",
    "text_mentions", # all prev turns + mentions
    "outtext_mentions",
    "lasttext",
    "lasttext_mentions", # last partner turn + mentions
    "lastlasttext_mentions", # last agent turn + mentions
    "prev_text_mentions", # everything up to your prev turn + mentions
    "partner_markers", # beginning of mention spliced with words
    "partner_amarkers", # beginning of numbered mention spliced with words
    "partner_tags", # bools for whether word is start of mention
    "partner_amentions",
)

tokenizer = get_bart_tokenizer()
unk_token = tokenizer.unk_token

domain = get_domain("one_common")
data = 'data/onecommon'
fold_num = 1
freq_cutoff = 10

# OPTIONS
"""
use_properties = True
use_pairwise_features = True
#use_pairwise_features = False
use_unordered_pairwise = True
#use_unordered_pairwise = False
use_extrema_features = False
use_unks = False
use_distance = True
use_confirm = True
use_select = True

# vary these only
use_short_describe = True
use_plan_specific_description = True
"""
# basic
basic_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgt,
    unordered_rel = True,
    short_describe = True,
    plan_specific_description = True,
    short_rel = False,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 7,
    dialog_history = True,
)

# plan limit + short rel
plan_limit_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgt,
    unordered_rel = True,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = True,
)

# plan limit + remove dialog
plan_limit_remove_dialog_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgt,
    unordered_rel = True,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
)

# plan limit + short_rel + ordered pairs
plan_limit_short_rel_ordered_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = True,
)

# plan limit + ordered pairs + group relations
plan_limit_ordered_group_rel_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = True,
)

# plan limit + group targets
plan_limit_group_target_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgts,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = True,
)

# plan limit + ordered pairs + group relations - dialog
plan_limit_ordered_group_rel_nodial_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
)

# plan limit + group targets - dialog
plan_limit_group_target_nodial_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgts,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
)

# ordered pairs + group relations - dialog
ordered_group_rel_nodial_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 7,
    dialog_history = False,
)

# group targets - dialog
group_target_nodial_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgts,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 7,
    dialog_history = False,
)

# plan limit + remove dialog
plan_limit_remove_dialog_unordered_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
)

plan_limit_ordered_group_rel_nodial_config_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
)

plan_limit_ordered_group_rel_nodial_config_agree_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
)

plan_limit_ordered_group_rel_nodial_config_agree_balance_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
    balance = True,
)

plan_limit_ordered_group_tgt_nodial_config_agree_balance_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelTgts,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
    balance = True,
)

mentiononly_planlimit_nodial_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
)

# 16
plan_limit_ordered_group_rel_nodial_config_agree_coref_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
)

# 17
mentiononly_planlimit_nodial_coref_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
)

# 18
mentiononly_planlimit_nodial_coref_balance_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = False,
    must_agree_config = True,
    balance = True,
    mention_specific_description = True,
)

# 19
mentiononly_planlimit_lastturn_coref_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = True,
    last_turn = True,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
)

# 20
mentiononly_planlimit_coref_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = True,
    last_turn = False,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
)

# 21
dot_lastturn_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = True,
    last_turn = True,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
    raw_dots = True,
)

# 22
dot_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 2,
    max_plan_size = 5,
    dialog_history = True,
    last_turn = False,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
    raw_dots = True,
)

# 23
large_dot_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 0,
    max_plan_size = 5,
    dialog_history = True,
    last_turn = False,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
    raw_dots = True,
)

# 24
all_plan_dot_lastturn_options = HfDataOptions(
    properties = [
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        #Property.RDIST,
    ],
    format = DescriptionFormat.SrcRelsTgt,
    unordered_rel = False,
    short_describe = True,
    plan_specific_description = True,
    short_rel = True,
    config_describe = True,
    confirmation = True,
    selection_leaning = True,
    selection = True,
    coref = True,
    min_plan_size = 0,
    max_plan_size = 5,
    dialog_history = True,
    last_turn = False,
    last_last_turn = True,
    must_agree_config = True,
    balance = False,
    mention_specific_description = True,
    raw_dots = True,
)


options = [
    basic_options, # 0
    plan_limit_options,
    plan_limit_remove_dialog_options, # 2
    plan_limit_short_rel_ordered_options,
    plan_limit_ordered_group_rel_options, # 4
    plan_limit_group_target_options,
    plan_limit_ordered_group_rel_nodial_options, # 6
    plan_limit_group_target_nodial_options,
    ordered_group_rel_nodial_options, # 8
    group_target_nodial_options,
    plan_limit_remove_dialog_unordered_options, # 10
    plan_limit_ordered_group_rel_nodial_config_options, # 11
    plan_limit_ordered_group_rel_nodial_config_agree_options, # 12
    plan_limit_ordered_group_rel_nodial_config_agree_balance_options, # 13
    plan_limit_ordered_group_tgt_nodial_config_agree_balance_options, # 14
    mentiononly_planlimit_nodial_options, # 15
    plan_limit_ordered_group_rel_nodial_config_agree_coref_options, # 16
    mentiononly_planlimit_nodial_coref_options, # 17
    mentiononly_planlimit_nodial_coref_balance_options, # 18
    mentiononly_planlimit_lastturn_coref_options, # 19
    mentiononly_planlimit_coref_options, # 20
    dot_lastturn_options, # 21
    dot_options, # 22
    large_dot_options, # 23
    all_plan_dot_lastturn_options , # 24
][24]
# for textgen: run 16 and 17 (+18)
# for mention gen: 19, 20
# for raw dots: 21
# for reference resolution: 23

dot_desc_template = Template(
    #"dot{{id}}: [x: {{x}}, y: {{y}}, size: {{size}}, color: {{color}}]"
    "dot{{id}} x: {{x}}, y: {{y}}, size: {{size}}, color: {{color}}"
)

corpus = ReferenceSentenceCorpus(
    domain, data,
    train='train_reference_{}.txt'.format(fold_num),
    valid='valid_reference_{}.txt'.format(fold_num),
    test='test_reference_{}.txt'.format(fold_num),
    freq_cutoff=freq_cutoff, verbose=True,
    max_instances_per_split=None,
    max_mentions_per_utterance=None,
    crosstalk_split=None,
    spatial_data_augmentation_on_train=False,
)

datadir = Path(data)

train_path = datadir / 'train_reference_{}.txt'.format(fold_num)
valid_path = datadir / 'valid_reference_{}.txt'.format(fold_num)
test_path = datadir / 'test_reference_{}.txt'.format(fold_num)

paths = {
    "train": train_path,
    "valid": valid_path,
    "test": test_path,
}
splits = paths.keys()

word_dict = Dictionary.from_file(str(train_path), freq_cutoff = freq_cutoff)

def splice_sentence_mentions(sent, refs, tokenizer):
    refs = np.array(refs).reshape(-1, 10)
    sent = [x for x in sent]
    original_sent = [x for x in sent]
    sent_markers = [x for x in sent]
    for idx, ref in reversed(list(zip(refs[:,0], refs[:,3:]))):
        sent.insert(idx, "<bom>")
        sent_markers.insert(idx, "<mention>")
        num_added = 1
        for i, val in enumerate(ref):
            if val:
                sent.insert(idx + num_added, f"dot{i+1}")
                num_added += 1
        sent.insert(idx+num_added, "<eom>")

    sent_amarkers = []
    # start count at 1
    count = 1
    for x in sent_markers:
        if x == "<mention>":
            sent_amarkers.append(f"<mention{count}>")
            count += 1
        else:
            sent_amarkers.append(x)

    # tags
    tokenized_sent = tokenizer.tokenize(" ".join(original_sent))
    tokenized_markers = tokenizer.tokenize(" ".join(sent_markers))
    tags = np.zeros(len(tokenized_sent), dtype=bool)
    if len(refs) > 0:
        # need to do an index search for each mention
        raw_mention_idxs = np.array([idx for idx, x in enumerate(tokenized_markers) if x == "<mention>"])
        mention_idxs = raw_mention_idxs - np.arange(len(raw_mention_idxs))
        #if mention_idxs.max() >= len(tokenized_sent):
            #import pdb; pdb.set_trace()
        tags[mention_idxs] = 1

        # dbg
        #print(" ".join(sent_markers))
        #print(np.array(tokenized_sent)[tags])

    return sent, sent_markers, tags.tolist(), sent_amarkers

"""
REFERENT LAYOUT: [
    beginning idx,
    ending idx,
    utterance end idx,
    *binary indicators for occurrence, eg 7 dots 0 1 1 0 0 0 0 => dots 1 and 2
]
"""
def _split_referents(words, referent_idxs, tokenizer):
    stops = ["YOU:", "THEM:"]
    sents_mentions = []
    sents_markers = []
    tagss = []
    sents_amarkers = []

    sents, current = [], []
    all_refs, current_refs = [], []
    split_ref_indices = []
    split_ref_objs = []
    assert len(referent_idxs) % 10 == 0
    for k in range(0, len(referent_idxs), 10):
        split_ref_indices.append(referent_idxs[k:k + 3])
        split_ref_objs.append(referent_idxs[k + 3:k + 10])
    split_ref_indices = np.array(split_ref_indices)
    ref_ix = 0
    for w in words:
        if w in stops:
            while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
                current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
                ref_ix += 1
            split_ref_indices[ref_ix:] -= len(current)
            if len(current) > 0:
                sents.append(current)
                all_refs.append(current_refs)
                sentref, sent_marker, tags, sent_amarker = splice_sentence_mentions(
                    current, current_refs, tokenizer)
                sents_mentions.append(sentref)
                sents_markers.append(sent_marker)
                sents_amarkers.append(sent_amarker)
                tagss.append(tags)
            current = []
            current_refs = []
        current.append(w)
    if len(current) > 0:
        while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
            current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
            ref_ix += 1
        sents.append(current)
        all_refs.append(current_refs)
        sentref, sent_marker, tags, sent_amarker = splice_sentence_mentions(
            current, current_refs, tokenizer)
        sents_mentions.append(sentref)
        sents_markers.append(sent_marker)
        sents_amarkers.append(sent_amarker)
        tagss.append(tags)
    assert ref_ix == len(split_ref_indices)
    assert sum(len(refs) for refs in all_refs) == len(referent_idxs)
    assert len(all_refs) == len(sents)
    return sents, all_refs, sents_mentions, sents_markers, tagss, sents_amarkers

class Conversation(NamedTuple):
    dots: list[float]
    sents: list[list[str]]
    refs: list[list[int]]
    partner_refs: list[list[int]]
    partner_refs_our_view: list[list[int]]

    scenario_id: str
    chat_id: str
    agent: int

    real_ids: list[str]
    partner_real_ids: list[str]

    # added, splices in mentions into sentences
    sentrefs: list[list[str]]
    partner_sentrefs: list[list[str]]
    partner_sentrefs_our_view: list[list[str]]
    all_sentrefs: list[list[str]]

    # added, splices beginning of mentions into sentences
    sentmarkers: list[list[str]]
    partner_sentmarkers: list[list[str]]
    all_sentmarkers: list[list[str]]
    all_sentamarkers: list[list[str]]

    # whether a word is the start of a mention
    tags: list[list[bool]]

def get_conversations(split, tokenizer):
    conversations = []
    path = paths[split]
    with path.open() as f:
        for line in f:
            line = line.strip()

            tokens = line.split()

            # The linearized context values
            # Each dot is represented as (x, y, size, color)
            # There should be 28 values = 4 values * 7 dots.
            input_vals = [float(val) for val in get_tag(tokens, 'input')]

            words = get_tag(tokens, 'dialogue')
            word_idxs = word_dict.w2i(words)

            referent_idxs = [int(val) for val in get_tag(tokens, 'referents')]
            partner_referent_idxs = [int(val) for val in get_tag(tokens, 'partner_referents')]
            partner_referent_our_view_idxs = [int(val) for val in get_tag(tokens, 'partner_referents_our_view')]

            # i believe this is the final selection? need to verify
            output_idx = int(get_tag(tokens, 'output')[0])

            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            partner_real_ids = get_tag(tokens, 'partner_real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]

            ref_disagreement = list(map(int, get_tag(tokens, 'referent_disagreements')))
            partner_ref_disagreement = list(map(int, get_tag(tokens, 'partner_referent_disagreements')))

            (
                sents, all_refs,
                sentrefs, sentmarkers, tags, sentamarkers,
            ) = _split_referents(words, referent_idxs, tokenizer)
            (
                sents_, all_partner_refs, partner_sentrefs,
                _, _, _,
            ) = _split_referents(words, partner_referent_idxs, tokenizer)
            assert sents == sents_
            (
                sents_, all_partner_refs_our_view, partner_sentrefs_our_view,
                partner_sentmarkers, partner_tags, partner_sentamarkers,
            ) = _split_referents(words, partner_referent_our_view_idxs, tokenizer)
            assert sents == sents_

            # merge YOU: and THEM:, all in our view
            all_sentrefs = [
                x if x[0] == "YOU:" else y
                for x,y in zip(sentrefs, partner_sentrefs_our_view)
            ]
            all_sentmarkers = [
                x if x[0] == "YOU:" else y
                for x,y in zip(sentmarkers, partner_sentmarkers)
            ]
            all_tags = [
                x if sents[i][0] == "YOU:" else y
                for i, (x,y) in enumerate(zip(tags, partner_tags))
            ]
            all_sentamarkers = [
                x if x[0] == "YOU:" else y
                for x,y in zip(sentamarkers, partner_sentamarkers)
            ]

            conversations.append(Conversation(
                #sents = sents if not use_unks else unk_sents(sents, word_dict),
                sents = sents,
                refs = all_refs,
                partner_refs = all_partner_refs,
                partner_refs_our_view = all_partner_refs_our_view,

                dots = input_vals,

                scenario_id = scenario_id,
                chat_id = chat_id,
                agent = agent,

                real_ids = real_ids,
                partner_real_ids = partner_real_ids,

                # sentences spliced with mentions
                sentrefs = sentrefs,
                partner_sentrefs = partner_sentrefs,
                partner_sentrefs_our_view = partner_sentrefs_our_view,
                all_sentrefs = all_sentrefs,

                # sentences spliced with mention MARKERS
                sentmarkers = sentmarkers,
                partner_sentmarkers = partner_sentmarkers,
                all_sentmarkers = all_sentmarkers,

                tags = all_tags,
                all_sentamarkers = all_sentamarkers,
            ))
        return conversations

def describe_dot_old(i, dot_strings, dots, size_color):
    rounded_dots = (dots.round(2) * 100).astype(int)
    num_dots = dots.shape[0]
    # (x, y, size, color)
    return dot_desc_template.render(
        id = i+1,
        x = rounded_dots[i, 0],
        y = rounded_dots[i, 1],
        size = rounded_dots[i, 2],
        color = rounded_dots[i, 3],
    )

def describe_dot(i, dot_strings, dots, size_color):
    size_map = template.size_map5
    color_map = template.color_map5
    size, color = size_color[i]
    return f"{dot_strings[i]} size {size_map[size]} and color {color_map[color]}"

def describe_dot_pair(
    i, j, dot_strings, dots,
    short=False, group_attributes=False,
):
    # does not use quantized properties
    dot1 = dot_strings[i]
    dot2 = dot_strings[j]
    x1, y1, s1, c1 = dots[i]
    x2, y2, s2, c2 = dots[j]

    # TODO: i think the y values are negated, so this needs to be flipped
    #vert_comp = "above" if y1 > y2 else "below"
    vert_comp = "above" if y1 < y2 else "below"
    hor_comp = "right" if x1 > x2 else "left"
    size_comp = "bigger" if s1 > s2 else "smaller"
    #col_comp = "darker" if c1 > c2 else "lighter"
    col_comp = "darker" if c1 < c2 else "lighter"

    if group_attributes:
        return f"{dot1} {vert_comp} {hor_comp} {size_comp} {col_comp} {dot2}"

    if not short:
        vert_str = f"{dot1} is {vert_comp} {dot2}"
        hor_str = f"{dot1} is {hor_comp} of {dot2}"
        size_str = f"{dot1} is {size_comp} than {dot2}"
        col_str = f"{dot1} is {col_comp} than {dot2}"
        return ", ".join([vert_str, hor_str, size_str, col_str])
    else:
        vert_str = f"{dot1} {vert_comp} {dot2}"
        hor_str = f"{dot1} {hor_comp} {dot2}"
        size_str = f"{dot1} {size_comp} {dot2}"
        col_str = f"{dot1} {col_comp} {dot2}"
        return ", ".join([vert_str, hor_str, size_str, col_str])

def get_relations(
    i, j, dots, eps=0.05,
):
    # does not use quantized properties
    x1, y1, s1, c1 = dots[i]
    x2, y2, s2, c2 = dots[j]

    # return binary vector with relations
    # [above, below, left, right, bigger, smaller, darker, lighter]
    #above = y1 > y2 + eps # flipped
    #below = y1 < y2 - eps # flipped
    below = y1 > y2 + eps
    above = y1 < y2 - eps
    left  = x1 < x2 - eps
    right = x1 > x2 + eps
    bigger = s1 > s2 + eps
    smaller = s1 < s2 - eps
    #darker = c1 > c2 + eps # flipped?
    #lighter = c1 < c2 - eps # flipped?
    lighter = c1 > c2 + eps
    darker = c1 < c2 - eps

    return np.array([
        above, below,
        left, right,
        bigger, smaller,
        darker, lighter,
    ], dtype=bool)

def describe_relations(relation_vector):
    strings = [
        "above", "below",
        "left", "right",
        "bigger", "smaller",
        "darker", "lighter",
    ]
    assert len(strings) == len(relation_vector)
    return " ".join([s for s,b in zip(strings, relation_vector) if b])

def describe_dot_tgts(
    i, js, dot_strings, dots,
):
    """
    Describe dot1 {relation} dot2s
    """
    # do not use quantized properties
    dot1 = dot_strings[i]
    dot2s = [dot_strings[j] for j in js]

    x1, y1, s1, c1 = dots[i]
    x2s = dots[js, 0]
    y2s = dots[js, 1]
    s2s = dots[js, 2]
    c2s = dots[js, 3]

    # TODO: i think the y values are negated, so this needs to be flipped
    aboves = y1 > y2s
    belows = y1 < y2s
    rights = x1 > x2s
    lefts = x1 < x2s
    biggers = s1 > s2s
    smallers = s1 < s2s
    darkers = c1 > c2s
    lighters =  c1 < c2s

    comps = []
    if aboves.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if aboves[i]])
        comps.append(f"{dot1} above {sdots}")
    if belows.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if belows[i]])
        comps.append(f"{dot1} below {sdots}")
    if rights.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if rights[i]])
        comps.append(f"{dot1} right {sdots}")
    if lefts.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if lefts[i]])
        comps.append(f"{dot1} left {sdots}")
    if biggers.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if biggers[i]])
        comps.append(f"{dot1} bigger {sdots}")
    if smallers.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if smallers[i]])
        comps.append(f"{dot1} smaller {sdots}")
    if darkers.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if darkers[i]])
        comps.append(f"{dot1} darker {sdots}")
    if lighters.sum() > 0:
        sdots = " ".join([x for i,x in enumerate(dot2s) if lighters[i]])
        comps.append(f"{dot1} lighter {sdots}")

    return ", ".join(comps)


def unit_vector(vector, axis=None):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector, axis=axis, keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_angles(xys):
    num_dots = xys.shape[0]
    pairs = [
        [tgt for tgt in range(num_dots) if src != tgt]
        for src in range(num_dots)
    ]
    xy_pairs = xys[np.array(pairs)]
    diffs = xys[:,None] - xy_pairs
    diffs = unit_vector(diffs, 1)
    return np.arccos(np.clip(
        (diffs[:,0] * diffs[:,1]).sum(-1),
        -1, 1
    ))

    # buggy?
    flat_diffs = diffs.reshape(-1, 2)
    return np.arccos(np.clip((
        flat_diffs[:,None] * flat_diffs
    ).sum(-1), -1., 1,))


def describe_dots(
    dots,
    use_short_describe = True,
    use_pairwise_features = True,
    use_unordered_pairwise = True,
    use_short_pairwise = True,
    use_config = True,
):
    dots = np.array(dots, dtype=float).reshape(-1, 4)
    dots[:,1] *= -1
    rounded_dots = (dots.round(2) * 100).astype(int)

    num_dots = dots.shape[0]
    dot_strings = [f"dot{i}" for i in range(1, num_dots+1)]

    """
    # (x, y, size, color)
    description_old = " [SEP] ".join([
        dot_desc_template.render(
            id = i+1,
            x = rounded_dots[i, 0],
            y = rounded_dots[i, 1],
            size = rounded_dots[i, 2],
            color = rounded_dots[i, 3],
        ) for i in range(num_dots)
    ])
    """

    ctx = process_ctx(dots)
    describe_dot_fn = describe_dot if use_short_describe else describe_dot_old

    descs = [describe_dot_fn(i, dot_strings, dots, ctx) for i in range(num_dots)]
    description = " [SEP] ".join(descs)

    if use_pairwise_features:
        # construct pairwise descriptions for each dot and 3 closest
        xy = dots[:,:2]
        dists = ((xy[:,None] - xy) ** 2).sum(-1)
        dists[range(7), range(7)] = dists.max() + 1
        closest_idxs = dists.argsort()[:,:2]

        # ordered pairs
        dot_pairs = [(i, j) for i in range(7) for j in closest_idxs[i]]
        if use_unordered_pairwise:
            # unordered pairs
            dot_pairs = set([tuple(sorted(x)) for x in dot_pairs])

        pairwise_strs = []
        for i,j in dot_pairs:
            pairwise_strs.append(describe_dot_pair(
                i, j, dot_strings, dots,
                short = use_short_pairwise,
            ))

        pairwise_str = ", ".join(pairwise_strs)
        description = f"{description} [SEP] {pairwise_str}"

    if use_config:
        pass

    return description


def describe_plan_specific_dots(
    dots,
    plan,
    use_unordered_pairwise = True,
    close_dots = None,
    use_short_pairwise = True,
    use_config = True,
    format = DescriptionFormat.SrcRelTgt,
):
    extras = None

    boolplan = plan.astype(bool)
    dots = np.array(dots, dtype=float).reshape(-1, 4)
    rounded_dots = (dots.round(2) * 100).astype(int)

    num_dots = dots.shape[0]
    dot_strings = [f"dot{i}" for i in range(1, num_dots+1)]

    ctx = process_ctx(dots)
    descs = [
        describe_dot(i, dot_strings, dots, ctx)
        for i in range(num_dots)
        if boolplan[i]
    ]
    description = " [SEP] ".join(descs)

    if use_config:
        # only run this in plan-specific, since it can get really slow
        num_dots = dots.shape[0]
        config_sizes = [2,3]
        #config_sizes = [3]
        config_descs = []
        triangle_configs = []
        line_configs = []
        for size in config_sizes:
            plan_dots = plan.nonzero()[0]
            combinations = list(itertools.combinations(plan_dots, size))
            for idxs in combinations:
                # dots: (x,y,size,color)
                config = dots[idxs,:]
                xy = config[:,:2]
                pairwise_dists = ((xy[:,None] - xy) ** 2).sum(-1)

                # describe_config(config, size)
                if size == 2:
                    # TODO: fold this into pairwise
                    dist = pairwise_dists[0,1]
                    # hard-coded threshold
                    if dist < 0.1:
                        config_descs.append(f"dot{str(idxs[0]+1)} close dot{str(idxs[1]+1)}")
                elif size == 3:
                    multihot = np.zeros(7, dtype=bool)
                    multihot[list(idxs)] = True
                    contig = is_contiguous(multihot, dots[:,:2], 7)
                    angles = get_angles(xy)
                    max_angle = angles.max() * 180 / math.pi
                    # hard-coded threshold
                    #if max_angle > 170 and contig:
                    if max_angle > 135:
                        config_descs.append(
                            f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                            "line"
                        )
                        line_configs.append(idxs)
                    elif max_angle <= 135 and contig:
                        config_descs.append(
                            f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                            "triangle"
                        )
                        triangle_configs.append(idxs)
        if len(config_descs) > 0:
            config_descriptions = " [SEP] ".join(config_descs)
            description = f"{description} [SEP] {config_descriptions}"
            extras = GenerationExtras(
                triangle_configs = triangle_configs,
                line_configs = line_configs,
            )

    if format == DescriptionFormat.SrcRelTgt or format == DescriptionFormat.SrcRelsTgt:
        # construct pairwise features for dot pairs in plan
        dot_pairs = [
            (i, j) for i in range(7) for j in range(7)
            if i != j and boolplan[i] and boolplan[j]
        ]
        if use_unordered_pairwise:
            # unordered pairs
            dot_pairs = set([tuple(sorted(x)) for x in dot_pairs])

        pairwise_strs = []
        for i,j in dot_pairs:
            pairwise_strs.append(describe_dot_pair(
                i, j,
                dot_strings,
                dots,
                short = use_short_pairwise,
                group_attributes = format == DescriptionFormat.SrcRelsTgt,
            ))
            if close_dots is not None:
                raise NotImplementedError("Need to implement distance")

        pairwise_str = " , ".join(pairwise_strs)
        description = f"{description} [SEP] {pairwise_str}"

        return description, extras
    elif format == DescriptionFormat.SrcRelTgts:
        ijs = [
            (i, [j for j in range(7) if boolplan[j] if i != j])
            for i in range(7) if boolplan[i]
        ]
        pairwise_strs = []
        for i, js in ijs:
            pairwise_strs.append(describe_dot_tgts(i, js, dot_strings, dots))
        pairwise_str = " , ".join(pairwise_strs)
        description = f"{description} [SEP] {pairwise_str}"
        return description, extras
    else:
        raise ValueError(f"Invalid format: {format.name}")

def describe_mention(idxs, dots):
    size = len(idxs)
    config = dots[idxs,:]
    xy = config[:,:2]
    pairwise_dists = ((xy[:,None] - xy) ** 2).sum(-1)

    # describe all shared properties?

    # describe_config(config, size)
    if size == 2:
        # TODO: fold this into pairwise
        dist = pairwise_dists[0,1]
        # hard-coded threshold
        if dist < 0.1:
            return f"dot{str(idxs[0]+1)} close dot{str(idxs[1]+1)}"
        else:
            return describe_set(idxs)
    elif size == 3:
        multihot = np.zeros(7, dtype=bool)
        multihot[list(idxs)] = True
        contig = is_contiguous(multihot, dots[:,:2], 7)
        angles = get_angles(xy)
        max_angle = angles.max() * 180 / math.pi
        # hard-coded threshold
        #if max_angle > 170 and contig:
        if max_angle > 135:
            return (
                f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                "line"
            )
        elif max_angle <= 135 and contig:
            return (
                f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                "triangle"
            )
        else:
            return describe_set(idxs)
    elif size == 4:
        multihot = np.zeros(7, dtype=bool)
        multihot[list(idxs)] = True
        contig = is_contiguous(multihot, dots[:,:2], 7)
        angles = get_angles(xy)
        max_angle = angles.max() * 180 / math.pi
        # hard-coded threshold
        #if max_angle > 170 and contig:
        if max_angle > 135:
            return (
                f"dot{str(idxs[0]+1)} dot{str(idxs[1]+1)} dot{str(idxs[2]+1)} "
                "line"
            )
            # TODO: add more configs later
        else:
            return describe_set(idxs)
    else:
        return describe_set(idxs)

def describe_set(dots):
    return " ".join(f"dot{d+1}" for d in dots)

def describe_mention_specific_dots(
    dots,
    plan,
    mentions,
    use_short_pairwise = True,
    use_config = True,
    format = DescriptionFormat.SrcRelTgt,
):
    extras = None

    boolplan = plan.astype(bool)
    boolmentions = mentions.astype(bool)
    dots = np.array(dots, dtype=float).reshape(-1, 4)
    rounded_dots = (dots.round(2) * 100).astype(int)

    num_dots = dots.shape[0]
    dot_strings = [f"dot{i}" for i in range(1, num_dots+1)]

    ctx = process_ctx(dots)
    # unary
    descs = [
        describe_dot(i, dot_strings, dots, ctx)
        for i in range(num_dots)
        if boolplan[i]
    ]
    description = " [SEP] ".join(descs)

    # TODO: replace unary dot descriptions for mention descriptions
    mentionsets = [set(m.nonzero()[0]) for m in mentions]
    if len(mentionsets) == 0:
        # no mentions to describe
        return "none"

    mention_descriptions = [
        describe_mention(tuple(mentionsets[0]), dots)
    ]

    for src_mention, tgt_mention in zip(mentionsets, mentionsets[1:]):
        src_str = describe_set(src_mention)
        tgt_str = describe_set(tgt_mention)

        src_diff = src_mention.difference(tgt_mention)

        # get relations from src_diff dots to tgt
        relation_intersection = None
        for src in src_diff:
            for tgt in tgt_mention:
                relation_set = get_relations(src, tgt, dots)
                relation_intersection = (
                    relation_set
                    if relation_intersection is None
                    else relation_intersection & relation_set
                )
        if relation_intersection is None:
            #mention_descriptions.append(f"{src_str} none {tgt_str}")
            mention_descriptions.append(f"none")
        else:
            relation_string = describe_relations(relation_intersection)
            mention_descriptions.append(relation_string)
            #mention_descriptions.append(
                #f"{src_str} {relation_string} {tgt_str}"
            #)
        mention_descriptions.append(describe_mention(tuple(tgt_mention), dots))

    mention_description = " [SEP] ".join(mention_descriptions)

    return f"{description} [MSEP] {mention_description}"



def describe_plan_dense(plan):
    num_dots = len(plan)
    dot_desc_template = Template("dot{{id}}: {{include}}")
    description = ", ".join([
        dot_desc_template.render(
            id = i+1,
            include = plan[i],
        ) for i in range(num_dots)
    ])
    return description

def describe_plan_sparse(plan):
    num_dots = len(plan)
    dot_desc_template = Template("dot{{id}}")
    description = " , ".join([
        f"dot{i+1}"
        for i in range(num_dots)
        if plan[i] == 1
    ])
    return description

# unused for now
class Example(NamedTuple):
    sents: list[list[str]]
    refs: list[list[int]]
    partner_refs: list[list[int]]
    partner_refs_our_view: list[list[int]]

    scenario_id: str
    chat_id: str

    real_ids: list[str]
    partner_real_ids: list[str]

    # new stuff
    text: str
    plan: list[int]
    mentions: list[int]

def unk_sents(sents, word_dict):
    unked_sents = [
        [word if word in word_dict.word2idx else unk_token for word in sent]
        for sent in sents
    ]
    return unked_sents

def get_confirmations(
    conversations,
    confirmation_tokenizer = None,
    confirmation_predictor = None,
):
    confirmations = {
        conversation.scenario_id: {} for conversation in conversations
    }
    for conversation in track(conversations):
    #for conversation in conversations:
        # mapping from conversation -> turn -> confirmation
        num_turns = len(conversation.sents)
        for turn in range(num_turns):
            is_you = conversation.sents[turn][0] == "YOU:"
            if is_you:
                # drop "YOU:" and "<eos>"
                sent = " ".join(conversation.sents[turn][1:-1])
                tokenized_text = confirmation_tokenizer(sent)
                response_struct = confirmation_predictor(
                    torch.tensor(tokenized_text["input_ids"])[None],
                )
                #response_logits = response_struct.logits[0].log_softmax(-1)
                #label = response_logits.argmax().item()
                confirmation_prediction = response_struct.logits[0].argmax().item()
                confirmations[conversation.scenario_id][str(turn)] = confirmation_prediction
                # need turn to be a str. after json deserialization keys turn into str.
    return confirmations

"""
Process conversations into examples.
Each conversation will be turned into num_turns examples (or num_turns / 2).
Each example will look like:
    * All previous turns: each turn is a list of tokens (strings)
    * All previous refs, partner refs, partner_refs_our_view
    * scenario, chat, and ids
Eg conversation prefixes
"""
def get_examples(
    conversations,
    confirmations,
    options,
    describe_plan = describe_plan_sparse,
):
    examples = {
        key: [] for key in fields
        #for key in Conversation._fields + fields
    }
    num_skipped = 0
    num_examples = 0
    num_skipped_agree_config = 0

    # triangle counting
    num_input_triangles = 0
    num_output_triangles = 0
    num_both_triangles = 0

    # line counting
    num_input_lines = 0
    num_output_lines = 0
    num_both_lines = 0

    num_input_triangle_output_line = 0 
    num_input_line_output_triangle = 0

    extra_counter = 0

    # plans of size 4+ are in the last bucket
    plan_sizes = [2,3,4]
    num_examples_by_plan_size = defaultdict(int)
    examples_by_plan_size = {
        size: {
            key: [] for key in fields
        } for size in plan_sizes
    }

    #for conversation in track(conversations):
    for conversation in conversations:
        num_turns = len(conversation.sents)

        dots = np.array(conversation.dots, dtype=float).reshape(-1, 4)
        xy = dots[:,:2]
        dists = ((xy[:,None] - xy[None]) ** 2).sum(-1)
        # lower is better
        dist_ranks = np.argsort(dists.flatten()).reshape(dists.shape)
        for turn in range(num_turns):
            # dont use Conversation as examples, want something that can be
            # directly fed to BART/RoBERTa
            """
            new_conversation = conversation._replace(
                sents = conversation.sents[:turn],
                refs = conversation.refs[:turn],
                partner_refs = conversation.partner_refs[:turn],
                partner_refs_our_view = conversation.partner_refs_our_view[:turn],
            )
            for key in Conversation._fields:
                examples[key].append(getattr(new_conversation, key))
            """
            raw_mentions = np.array(conversation.refs[turn]).reshape((-1, 10))[:,3:]
            raw_plan = raw_mentions.any(0).astype(int)

            partner_refs = conversation.partner_refs_our_view[turn-1] if turn > 0 else []
            raw_partner_mentions = (
                np.array(partner_refs).reshape((-1, 10))[:,3:]
                if len(partner_refs) > 0
                else None
            )

            if options.max_plan_size > 0 and raw_plan.sum() > options.max_plan_size:
                num_skipped += 1
                continue

            if options.min_plan_size > 0 and raw_plan.sum() < options.min_plan_size:
                num_skipped += 1
                continue

            is_you = conversation.sents[turn][0] == "YOU:"

            # compare partner_markers to raw_partner_mentions
            ltext = conversation.all_sentmarkers[turn-1] if turn > 0 else "none"
            num_men = sum([1 for w in ltext if w == "<mention>"])
            lenmen = len(raw_partner_mentions) if raw_partner_mentions is not None else 0
            if is_you and num_men != lenmen:
                num_skipped += 1
                continue

            if is_you:
                # mention-specific dot-representations
                mention_description = describe_mention_specific_dots(
                    conversation.dots,
                    raw_plan,
                    raw_mentions,
                    use_short_pairwise = options.short_rel,
                    use_config = options.config_describe,
                    format = options.format,
                )
                # plan-specific dot representations
                dot_description, generation_extras = describe_plan_specific_dots(
                    conversation.dots,
                    raw_plan,
                    options.unordered_rel,
                    close_dots = None,
                    use_short_pairwise = options.short_rel,
                    use_config = options.config_describe,
                    format = options.format,
                )
                # output sentence
                sent = " ".join(conversation.sents[turn])

                # check if heuristics all agree
                triangle_in_input = "triangle" in dot_description
                triangle_in_output = "triangle" in sent

                line_in_input = "line" in dot_description
                line_in_output = "line" in sent

                num_input_triangles += triangle_in_input
                num_output_triangles += triangle_in_output
                num_both_triangles += triangle_in_input & triangle_in_output

                num_input_lines +=  line_in_input
                num_output_lines += line_in_output
                num_both_lines += line_in_input & line_in_output

                num_input_triangle_output_line += triangle_in_input & line_in_output
                num_input_line_output_triangle += line_in_input & triangle_in_output

                if options.must_agree_config:
                    # output should be explained by something
                    triangle_not_explained = (not triangle_in_input) & triangle_in_output
                    line_not_explained = (not line_in_input) & line_in_output
                    if triangle_not_explained or line_not_explained:
                        # skip this example
                        num_skipped_agree_config += 1
                        continue

                if False:
                #if triangle_in_input & (not triangle_in_output):
                #if line_in_input & (not line_in_output):
                #if line_in_input & line_in_output:
                #if (not line_in_input) & line_in_output:
                    boolplan = raw_plan.astype(bool)
                    extra_counter += 1
                    print(f"num examples: {extra_counter}")
                    print(generation_extras)
                    print(conversation.scenario_id)
                    print(conversation.real_ids)
                    print(examples["outtext_mentions"][-1])

                    dots = np.array(conversation.dots, dtype=float).reshape(-1, 4)
                    xys = dots[boolplan, :2]
                    angles = get_angles(xys) * 180 / math.pi
                    angle = angles.max()
                    vectors = xys[:,None] - xys

                # conversation metadata
                examples["chat_id"].append(conversation.chat_id)
                examples["scenario_id"].append(conversation.scenario_id)
                examples["agent"].append(conversation.agent)

                # dots
                examples["raw_dots"].append(conversation.dots)

                # textify all dot properties

                # mention = (start idx, end idx, utterance end idx, *binary ind for 7 dots)
                examples["plan"].append(describe_plan(raw_plan))
                examples["mentions"].append(
                    " [SEP] ".join([describe_plan(m) for m in raw_mentions])
                )
                examples["raw_mentions"].append(
                    raw_mentions.tolist()
                    if raw_mentions is not None
                    else []
                )
                examples["joint_mentions"].append(
                    bitutils.config_to_int(raw_mentions.astype(bool)).tolist()
                )
                examples["partner_mentions"].append(
                    " [SEP] ".join([describe_plan(m) for m in raw_partner_mentions])
                    if raw_partner_mentions is not None
                    else "none"
                )
                examples["raw_partner_mentions"].append(
                    raw_partner_mentions.tolist()
                    if raw_partner_mentions is not None
                    else []
                )
                examples["joint_partner_mentions"].append(
                    np.packbits(
                        np.flip(raw_partner_mentions, -1),
                        axis = -1,
                        bitorder = "little",
                    )[:,0].tolist()
                    if raw_partner_mentions is not None
                    else []
                )
                examples["partner_amentions"].append(
                    " [SEP] ".join([
                        f"<mention{i+1}> {describe_plan(m)}"
                        for i, m in enumerate(raw_partner_mentions)
                    ])
                    if raw_partner_mentions is not None
                    else "none"
                )

                # linearized dot representation
                examples["dots"].append(describe_dots(
                    conversation.dots,
                    use_short_describe = options.short_describe,
                    use_pairwise_features = True,
                    use_unordered_pairwise = options.unordered_rel,
                    use_short_pairwise = options.short_rel,
                    use_config = options.config_describe,
                ))

                examples["mention_specific_dots"].append(mention_description)
                examples["plan_specific_dots"].append(dot_description)

                # concatenate all text
                examples["text"].append(
                    " ".join([x for xs in conversation.sents[:turn] for x in xs])
                )

                examples["outtext"].append(sent)

                examples["lasttext"].append(
                    " ".join(conversation.sents[turn-1])
                    if turn > 0
                    else "first"
                )

                # confirmation
                confirmation_prediction = confirmations[conversation.scenario_id][str(turn)]
                # 0: None, 1: Confirm, 2: Disconfirm
                confirm_map = ["none", "yes", "no"]
                examples["confirmation"].append(
                    f"confirmation: {confirm_map[confirmation_prediction]}"
                )

                # selection-leaning
                selection_like_words = set(["pick", "choose", "select", "click"])
                has_select = any([
                    x in selection_like_words for x in conversation.sents[turn]
                ])
                examples["selection_leaning"].append(
                    "pick"
                    if has_select
                    else "no pick"
                )

                # selection
                examples["selection"].append(
                    "<selection>"
                    if turn == num_turns - 1
                    else "no select"
                )

                # pronoun
                pronoun_words = set(["it", "them", "that", "those", "your", "they"])
                has_pronoun = any([
                    x in pronoun_words for x in conversation.sents[turn]
                ])
                if options.coref:
                    examples["coref"].append(
                        "coref"
                        if has_pronoun
                        else "no coref"
                    )
                else:
                    examples["coref"].append("none")

                # sentrefs
                examples["outtext_mentions"].append(
                    " ".join(conversation.sentrefs[turn])
                )
                examples["lasttext_mentions"].append(
                    " ".join(conversation.all_sentrefs[turn-1])
                    if turn > 0
                    else "first"
                )
                # your prev utt, before partner's
                examples["lastlasttext_mentions"].append(
                    " ".join(conversation.all_sentrefs[turn-2])
                    if turn > 1
                    else "first"
                )
                examples["text_mentions"].append(
                    " ".join([x for xs in conversation.all_sentrefs[:turn] for x in xs])
                    if turn > 0 else "none"
                )
                examples["prev_text_mentions"].append(
                    " ".join([x for xs in conversation.all_sentrefs[:turn-1] for x in xs])
                    if turn > 1
                    else "none"
                )

                examples["markers"].append(
                    " ".join(conversation.all_sentmarkers[turn])
                )
                examples["amarkers"].append(
                    " ".join(conversation.all_sentamarkers[turn])
                )
                examples["tags"].append(
                    conversation.tags[turn]
                )

                examples["partner_markers"].append(
                    " ".join(conversation.all_sentmarkers[turn-1])
                    if turn > 0 else "none"
                )
                examples["partner_amarkers"].append(
                    " ".join(conversation.all_sentamarkers[turn-1])
                    if turn > 0 else "none"
                )
                examples["partner_tags"].append(
                    conversation.tags[turn-1]
                    if turn > 0 else []
                )

                num_examples += 1

                # copy to examples by plan size
                plan_size = int(raw_plan.sum())
                if plan_size > 4:
                    # round down to 4
                    plan_size = 4
                num_examples_by_plan_size[plan_size] += 1
                if plan_size in examples_by_plan_size:
                    for field in examples.keys():
                        examples_by_plan_size[plan_size][field].append(examples[field][-1])

            #for field in examples.keys():
                #print(field, examples[field][-1])
            #import pdb; pdb.set_trace()


    print(f"num examples {num_examples}")
    print(f"Number of examples skipped due to plan size: {num_skipped}")
    print(f"Input triangles: {num_input_triangles} || output triangles: {num_output_triangles} || both triangles: {num_both_triangles}")
    print(f"Input lines: {num_input_lines} || output lines: {num_output_lines} || both lines: {num_both_lines}")
    print(f"triangle but line {num_input_triangle_output_line}")
    print(f"line but triangle {num_input_line_output_triangle}")
    print(f"number of examples skipped due to unexplained config: {num_skipped_agree_config}")
    print(num_examples_by_plan_size)

    if options.balance:
        # rebalance examples by plan size
        num_exs = [num_examples_by_plan_size[size] for size in plan_sizes]
        largest = max(num_exs)
        multipliers = largest // np.array(num_exs)
        new_examples = {}
        for field in examples.keys():
            new_examples[field] = [
                example
                for size, multiplier in zip(plan_sizes, multipliers)
                for example in examples_by_plan_size[size][field] * multiplier
            ]

        """
        # test on training only?
        for size, mult in zip(plan_sizes, multipliers):
            idx = 136
            field = "outtext"
            extext = examples_by_plan_size[size][field][idx]
            # check that number of examples matches multiplier
            num_app = 0
            for ex in new_examples[field]:
                if ex == extext:
                    num_app += 1
            assert mult == num_app

        # check that the fields are correctly paired
        idxs = [129, 513, 1039, 5000, 10000]
        for idx in idxs:
            extext = new_examples["outtext"][idx]
            for exidx in range(len(examples["outtext"])):
                if examples["outtext"][exidx] == extext:
                    for field in fields:
                        assert new_examples[field][idx] == examples[field][exidx]
        """

        examples = new_examples

    # Check number of examples for each field
    for key1 in fields:
        for key2 in fields:
            assert len(examples[key1]) == len(examples[key2])

    return examples

if __name__ == "__main__":
    redo_examples = True
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
    )
    import hfutils

    print(Conversation._fields)

    response_pretrained_path = "models/save_pretrained"
    confirmation_tokenizer = AutoTokenizer.from_pretrained(response_pretrained_path)
    confirmation_predictor = AutoModelForSequenceClassification.from_pretrained(
        response_pretrained_path)
    tokenizer = hfutils.get_bart_tokenizer()

    for split in splits:
        conversations = get_conversations(split, tokenizer)

        feature_string = construct_feature_string(options)

        confirmation_path = Path(f"hf_datasets/{split}_confirmations.json")
        confirmations = None
        if not confirmation_path.exists():
            print("Generating confirmations")
            confirmations = get_confirmations(
                conversations,
                confirmation_tokenizer = confirmation_tokenizer,
                confirmation_predictor = confirmation_predictor,
            )
            print(f"Saving confirmations to {str(confirmation_path)}")
            with confirmation_path.open("w") as f:
                json.dump(confirmations, f)
        else:
            print(f"Loading confirmations from {str(confirmation_path)}")
            with confirmation_path.open("r") as f:
                confirmations = json.load(f)

        # json path for saving examples
        json_path = Path(f"hf_datasets/{split}_{feature_string}.json")
        examples = None
        if not json_path.exists() or redo_examples:
            print("Generating new examples")
            examples = get_examples(
                conversations,
                confirmations,
                options,
            )
            print(f"Saving examples to {str(json_path)}")
            with json_path.open("w") as f:
                json.dump(examples, f)
        else:
            print(f"Loading examples from {str(json_path)}")
            with json_path.open("r") as f:
                examples = json.load(f)

        idx = 11
        print(f"Data example in {split}")
        for field in examples.keys():
            print(field, examples[field][idx])


        dot_descs = examples["dots"]
        if options.plan_specific_description:
            dot_descs = examples["plan_specific_dots"]
        if options.mention_specific_description:
            dot_descs = examples["mention_specific_dots"]

        # mention | plan, text gen
        num_examples = len(examples["mentions"])
        mention_given_text_plan_examples = {}
        mention_given_text_plan_examples["input"] = [
            f"{dots} [MSEP] {text} [MSEP] {plan}"
            for dots, text, plan in zip(
                examples["plan_specific_dots"],
                examples["text"] if not options.last_turn else examples["lasttext"],
                examples["plan"],
            )
        ]

        # add metadata
        mention_given_text_plan_examples["chat_id"] = examples["chat_id"]
        mention_given_text_plan_examples["scenario_id"] = examples["scenario_id"]
        mention_given_text_plan_examples["agent"] = examples["agent"]

        mention_given_text_plan_examples["label"] = examples["mentions"]
        mention_given_text_plan_dataset = Dataset.from_dict(mention_given_text_plan_examples)
        mention_given_text_plan_path = f"hf_datasets/{split}_mentions_given_text_plan_{feature_string}.hf"
        print(f"Mention dataset path {mention_given_text_plan_path}")
        mention_given_text_plan_dataset.save_to_disk(mention_given_text_plan_path)

        input_lens = [len(tokenizer.tokenize(x)) for x in mention_given_text_plan_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of mention|text input: {max_length_input}")
        print(mention_given_text_plan_examples["input"][np.argmax(input_lens)])

        output_lens = [len(tokenizer.tokenize(x)) for x in mention_given_text_plan_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of mention|text output: {max_length_output}")
        print(mention_given_text_plan_examples["label"][np.argmax(output_lens)])

        # mention | plan, textmention gen
        num_examples = len(examples["mentions"])
        mention_given_textmention_plan_examples = {}
        mention_given_textmention_plan_examples["input"] = [
            f"{dots} [MSEP] {text} [MSEP] {plan}"
            for dots, text, plan in zip(
                examples["plan_specific_dots"],
                examples["text_mentions"] if not options.last_turn else examples["lasttext_mentions"],
                examples["plan"],
            )
        ]

        # add metadata
        mention_given_textmention_plan_examples["chat_id"] = examples["chat_id"]
        mention_given_textmention_plan_examples["scenario_id"] = examples["scenario_id"]
        mention_given_textmention_plan_examples["agent"] = examples["agent"]

        mention_given_textmention_plan_examples["label"] = examples["mentions"]
        mention_given_textmention_plan_dataset = Dataset.from_dict(mention_given_textmention_plan_examples)
        mention_given_textmention_plan_path = f"hf_datasets/{split}_mentions_given_textmention_plan_{feature_string}.hf"
        print(f"Mention|textmention,plan dataset path {mention_given_textmention_plan_path}")
        mention_given_textmention_plan_dataset.save_to_disk(mention_given_textmention_plan_path)

        input_lens = [len(tokenizer.tokenize(x)) for x in mention_given_textmention_plan_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of mention|textmention input: {max_length_input}")
        print(mention_given_textmention_plan_examples["input"][np.argmax(input_lens)])

        output_lens = [len(tokenizer.tokenize(x)) for x in mention_given_textmention_plan_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of mention|textmention output: {max_length_output}")
        print(mention_given_textmention_plan_examples["label"][np.argmax(output_lens)])

        # mention | plan, text, consel gen
        num_examples = len(examples["mentions"])
        mention_given_text_plan_consel_examples = {}
        mention_given_text_plan_consel_examples["dots"] = examples["raw_dots"]
        if options.raw_dots:
            mention_given_text_plan_consel_examples["input"] = [
                f"{confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {coref} "
                f"[MSEP] {text} [MSEP] {plan}"
                for confirm, selection_leaning, selection, coref, dots, text, plan in zip(
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["coref"],
                    examples["plan_specific_dots"],
                    examples["text"] if not options.last_turn else examples["lasttext"],
                    examples["plan"],
                )
            ]
        else:
            mention_given_text_plan_consel_examples["input"] = [
                f"{confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {coref} "
                f"[MSEP] {dots} [MSEP] {text} [MSEP] {plan}"
                for confirm, selection_leaning, selection, coref, dots, text, plan in zip(
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["coref"],
                    examples["plan_specific_dots"],
                    examples["text"] if not options.last_turn else examples["lasttext"],
                    examples["plan"],
                )
            ]

        # add metadata
        mention_given_text_plan_consel_examples["chat_id"] = examples["chat_id"]
        mention_given_text_plan_consel_examples["scenario_id"] = examples["scenario_id"]
        mention_given_text_plan_consel_examples["agent"] = examples["agent"]

        mention_given_text_plan_consel_examples["label"] = examples["mentions"]
        mention_given_text_plan_consel_dataset = Dataset.from_dict(mention_given_text_plan_consel_examples)
        mention_given_text_plan_consel_path = f"hf_datasets/{split}_mentions_given_text_plan_consel_{feature_string}.hf"
        print(f"Mention|text,plan,x dataset path {mention_given_text_plan_consel_path}")
        mention_given_text_plan_consel_dataset.save_to_disk(mention_given_text_plan_consel_path)

        input_lens = [len(tokenizer.tokenize(x)) for x in mention_given_text_plan_consel_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of mention|text,X input: {max_length_input}")
        print(mention_given_text_plan_consel_examples["input"][np.argmax(input_lens)])

        output_lens = [len(tokenizer.tokenize(x)) for x in mention_given_text_plan_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of mention|text,X output: {max_length_output}")
        print(mention_given_text_plan_consel_examples["label"][np.argmax(output_lens)])


        # mention | plan, textmention, consel gen
        num_examples = len(examples["mentions"])
        mention_given_textmention_plan_consel_examples = {}
        mention_given_textmention_plan_consel_examples["dots"] = examples["raw_dots"]
        if options.raw_dots:
            mention_given_textmention_plan_consel_examples["input"] = [
                f"{confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {coref} "
                f"[MSEP] {text} [MSEP] {plan}"
                for confirm, selection_leaning, selection, coref, dots, text, plan in zip(
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["coref"],
                    examples["plan_specific_dots"],
                    examples["text_mentions"] if not options.last_turn else examples["lasttext_mentions"],
                    examples["plan"],
                )
            ]
        else:
            mention_given_textmention_plan_consel_examples["input"] = [
                f"{confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {coref} "
                f"[MSEP] {dots} [MSEP] {text} [MSEP] {plan}"
                for confirm, selection_leaning, selection, coref, dots, text, plan in zip(
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["coref"],
                    examples["plan_specific_dots"],
                    examples["text_mentions"] if not options.last_turn else examples["lasttext_mentions"],
                    examples["plan"],
                )
            ]

        # add metadata
        mention_given_textmention_plan_consel_examples["chat_id"] = examples["chat_id"]
        mention_given_textmention_plan_consel_examples["scenario_id"] = examples["scenario_id"]
        mention_given_textmention_plan_consel_examples["agent"] = examples["agent"]

        mention_given_textmention_plan_consel_examples["label"] = examples["mentions"]
        mention_given_textmention_plan_consel_dataset = Dataset.from_dict(mention_given_textmention_plan_consel_examples)
        mention_given_textmention_plan_consel_path = f"hf_datasets/{split}_mentions_given_textmention_plan_consel_{feature_string}.hf"
        print(f"Mention|textmention,plan,x dataset path {mention_given_textmention_plan_consel_path}")
        mention_given_textmention_plan_consel_dataset.save_to_disk(mention_given_textmention_plan_consel_path)

        input_lens = [len(tokenizer.tokenize(x)) for x in mention_given_textmention_plan_consel_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of mention|textmention,X input: {max_length_input}")
        print(mention_given_textmention_plan_consel_examples["input"][np.argmax(input_lens)])

        output_lens = [len(tokenizer.tokenize(x)) for x in mention_given_textmention_plan_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of mention|textmention,X output: {max_length_output}")
        print(mention_given_textmention_plan_consel_examples["label"][np.argmax(output_lens)])

        # plan gen
        num_examples = len(examples["plan"])
        plan_examples = {}
        #plan_examples["input"] = examples["text"]
        plan_examples["input"] = [
            f"{dots} [MSEP] {text}"
            for dots, text in zip(
                examples["dots"],
                examples["text"],
            )
        ]
        # not allowed to use plan-specific dot descriptions
        input_lens = [len(tokenizer.tokenize(x)) for x in plan_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of plan input: {max_length_input}")
        print(plan_examples["input"][np.argmax(input_lens)])

        plan_examples["label"] = examples["plan"]
        max_length_output = max([len(tokenizer.tokenize(x)) for x in plan_examples["label"]])
        print(f"Max length of plan output: {max_length_output}")
        plan_dataset = Dataset.from_dict(plan_examples)
        plan_path = f"hf_datasets/{split}_plan_given_text_{feature_string}.hf"
        print(f"Plan dataset path {plan_path}")
        plan_dataset.save_to_disk(plan_path)

        # text gen
        num_examples = len(examples["outtext"])
        text_examples = {}
        text_examples["input"] = [
            f"{dots} [MSEP] {text} [MSEP] {confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {plan}"
            for (
                dots,
                text,
                confirm,
                selection_leaning,
                selection,
                plan,
            ) in zip(
                dot_descs,
                examples["text"],
                examples["confirmation"],
                examples["selection_leaning"],
                examples["selection"],
                examples["plan"],
            )
        ]
        if not options.dialog_history:
            text_examples["input"] = [
                f"{dots} [MSEP] {confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {plan}"
                for (
                    dots,
                    text,
                    confirm,
                    selection_leaning,
                    selection,
                    plan,
                ) in zip(
                    dot_descs,
                    examples["text"],
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["plan"],
                )
            ]
        input_lens = [len(tokenizer.tokenize(x)) for x in text_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of text input: {max_length_input}")
        print(text_examples["input"][np.argmax(input_lens)])

        text_examples["label"] = examples["outtext"]
        max_length_output = max([len(tokenizer.tokenize(x)) for x in text_examples["label"]])
        print(f"Max length of text output: {max_length_output}")
        text_dataset = Dataset.from_dict(text_examples)
        text_path = f"hf_datasets/{split}_text_given_plan_{feature_string}.hf"
        print(f"Text dataset path {text_path}")
        text_dataset.save_to_disk(text_path)
        print(f"num text examples: {len(text_examples['input'])}")

        # text, mention gen
        num_examples = len(examples["outtext"])
        text_mention_examples = {}
        if options.dialog_history:
            text_mention_examples["input"] = [
                f"{dots} [MSEP] {text} [MSEP] {confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {plan}"
                for (
                    dots,
                    text,
                    confirm,
                    selection_leaning,
                    selection,
                    plan,
                ) in zip(
                    dot_descs,
                    examples["text"],
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["plan"],
                )
            ]
        elif not options.dialog_history:
            text_mention_examples["input"] = [
                f"{dots} [MSEP] {confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {plan}"
                for (
                    dots,
                    text,
                    confirm,
                    selection_leaning,
                    selection,
                    plan,
                ) in zip(
                    dot_descs,
                    examples["text"],
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["plan"],
                )
            ]
        input_lens = [len(tokenizer.tokenize(x)) for x in text_mention_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of text,mention input: {max_length_input}")
        print(text_mention_examples["input"][np.argmax(input_lens)])

        text_mention_examples["label"] = [
            f"{mentions} [MSEP] {outtext}"
            for outtext, mentions in zip(examples["outtext"], examples["mentions"])
        ]
        max_length_output = max([len(tokenizer.tokenize(x)) for x in text_mention_examples["label"]])
        print(f"Max length of text,mention output: {max_length_output}")
        text_mention_dataset = Dataset.from_dict(text_mention_examples)
        text_mention_path = f"hf_datasets/{split}_text_mention_given_plan_{feature_string}.hf"
        print(f"Text,mention dataset path {text_mention_path}")
        text_mention_dataset.save_to_disk(text_mention_path)
        print(f"num text,mention examples: {len(text_examples['input'])}")

        # textmention, mention gen
        num_examples = len(examples["outtext_mentions"])
        textmention_mention_examples = {}
        textmention_mention_examples["input"] = [
            f"{dots} [MSEP] {text} [MSEP] {confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {plan}"
            for (
                dots,
                text,
                confirm,
                selection_leaning,
                selection,
                plan,
            ) in zip(
                dot_descs,
                examples["text_mentions"],
                examples["confirmation"],
                examples["selection_leaning"],
                examples["selection"],
                examples["plan"],
            )
        ]
        if not options.dialog_history:
            textmention_mention_examples["input"] = [
                f"{dots} [MSEP] {confirm} [MSEP] {selection_leaning} [MSEP] {selection} [MSEP] {plan}"
                for (
                    dots,
                    text,
                    confirm,
                    selection_leaning,
                    selection,
                    plan,
                ) in zip(
                    dot_descs,
                    examples["text_mentions"],
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["plan"],
                )
            ]
        input_lens = [len(tokenizer.tokenize(x)) for x in textmention_mention_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of textmention,mention input: {max_length_input}")
        print(textmention_mention_examples["input"][np.argmax(input_lens)])

        textmention_mention_examples["label"] = [
            f"{mentions} [MSEP] {outtext}"
            for outtext, mentions in zip(examples["outtext_mentions"], examples["mentions"])
        ]
        output_lens = [len(tokenizer.tokenize(x)) for x in textmention_mention_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of textmention,mention output: {max_length_output}")
        print(textmention_mention_examples["label"][np.argmax(output_lens)])

        # add metadata
        textmention_mention_examples["chat_id"] = examples["chat_id"]
        textmention_mention_examples["scenario_id"] = examples["scenario_id"]
        textmention_mention_examples["agent"] = examples["agent"]

        textmention_mention_dataset = Dataset.from_dict(textmention_mention_examples)
        textmention_mention_path = f"hf_datasets/{split}_textmention_mention_given_plan_{feature_string}.hf"
        print(f"Textmention,mention dataset path {textmention_mention_path}")
        textmention_mention_dataset.save_to_disk(textmention_mention_path)
        print(f"num textmention,mention examples: {len(textmention_mention_examples['input'])}")

        # textmention | mention, context
        # use ground truth mentions
        num_examples = len(examples["outtext_mentions"])
        textmention_examples = {}
        textmention_examples["input"] = [
            f"{dots} [MSEP] {text} [MSEP] {confirm} [MSEP] "
            f"{selection_leaning} [MSEP] {selection} [MSEP] "
            f"{coref} [MSEP] "
            f"{mentions}"
            for (
                dots,
                text,
                confirm,
                selection_leaning,
                selection,
                coref,
                plan,
                mentions,
            ) in zip(
                dot_descs,
                examples["text_mentions"],
                examples["confirmation"],
                examples["selection_leaning"],
                examples["selection"],
                examples["coref"],
                examples["plan"],
                examples["mentions"],
            )
        ]
        if not options.dialog_history:
            textmention_examples["input"] = [
                f"{dots} [MSEP] {confirm} [MSEP] "
                f"{selection_leaning} [MSEP] {selection} [MSEP] "
                f"{coref} [MSEP] "
                f"{mentions}"
                for (
                    dots,
                    text,
                    confirm,
                    selection_leaning,
                    selection,
                    coref,
                    plan,
                    mentions,
                ) in zip(
                    dot_descs,
                    examples["text_mentions"],
                    examples["confirmation"],
                    examples["selection_leaning"],
                    examples["selection"],
                    examples["coref"],
                    examples["plan"],
                    examples["mentions"],
                )
            ]
        input_lens = [len(tokenizer.tokenize(x)) for x in textmention_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of textmention input: {max_length_input}")
        print(textmention_examples["input"][np.argmax(input_lens)])

        textmention_examples["label"] = examples["outtext_mentions"]
        output_lens = [len(tokenizer.tokenize(x)) for x in textmention_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of textmention output: {max_length_output}")
        print(textmention_examples["label"][np.argmax(output_lens)])

        # add metadata
        textmention_examples["chat_id"] = examples["chat_id"]
        textmention_examples["scenario_id"] = examples["scenario_id"]
        textmention_examples["agent"] = examples["agent"]

        textmention_dataset = Dataset.from_dict(textmention_examples)
        textmention_path = f"hf_datasets/{split}_textmention_given_mention_{feature_string}.hf"
        print(f"Textmention dataset path {textmention_path}")
        textmention_dataset.save_to_disk(textmention_path)
        print(f"num textmention examples: {len(textmention_examples['input'])}")

        # partner textmention | partner text, textmention history, dots
        # use ground truth mentions
        num_examples = len(examples["lasttext_mentions"])
        lasttext_mentions_examples = {}
        lasttext_mentions_examples["input"] = [
            f"{text} [MSEP] {lasttext}"
            for (
                text,
                lasttext,
            ) in zip(
                examples["prev_text_mentions"],
                examples["lasttext"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in lasttext_mentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of lasttext_mentions input: {max_length_input}")
        print(lasttext_mentions_examples["input"][np.argmax(input_lens)])

        lasttext_mentions_examples["label"] = examples["lasttext_mentions"]
        output_lens = [len(tokenizer.tokenize(x)) for x in lasttext_mentions_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of lasttext_mentions output: {max_length_output}")
        print(lasttext_mentions_examples["label"][np.argmax(output_lens)])

        # add metadata
        lasttext_mentions_examples["chat_id"] = examples["chat_id"]
        lasttext_mentions_examples["scenario_id"] = examples["scenario_id"]
        lasttext_mentions_examples["agent"] = examples["agent"]
        lasttext_mentions_examples["dots"] = examples["raw_dots"]

        lasttext_mentions_dataset = Dataset.from_dict(lasttext_mentions_examples)
        lasttext_mentions_path = f"hf_datasets/{split}_lasttext_mentions_{feature_string}.hf"
        print(f"lasttext_mentions dataset path {lasttext_mentions_path}")
        lasttext_mentions_dataset.save_to_disk(lasttext_mentions_path)
        print(f"num lasttext_mentions examples: {len(lasttext_mentions_examples['input'])}")

        # partner mention | partner markers, textmention history, dots
        # use ground truth mentions
        num_examples = len(examples["partner_mentions"])
        partner_mentions_examples = {}
        partner_mentions_examples["input"] = [
            f"{text} [MSEP] {lasttext}"
            for (
                text,
                lasttext,
            ) in zip(
                examples["prev_text_mentions"]
                    if not options.last_last_turn
                    else examples["lastlasttext_mentions"],
                examples["partner_markers"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in partner_mentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of partner_mentions input: {max_length_input}")
        print(partner_mentions_examples["input"][np.argmax(input_lens)])

        partner_mentions_examples["label"] = examples["partner_mentions"]
        output_lens = [len(tokenizer.tokenize(x)) for x in partner_mentions_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of partner_mentions output: {max_length_output}")
        print(partner_mentions_examples["label"][np.argmax(output_lens)])

        # add metadata
        partner_mentions_examples["chat_id"] = examples["chat_id"]
        partner_mentions_examples["scenario_id"] = examples["scenario_id"]
        partner_mentions_examples["agent"] = examples["agent"]
        partner_mentions_examples["dots"] = examples["raw_dots"]

        partner_mentions_dataset = Dataset.from_dict(partner_mentions_examples)
        partner_mentions_path = f"hf_datasets/{split}_partner_mentions_{feature_string}.hf"
        print(f"partner_mentions dataset path {partner_mentions_path}")
        partner_mentions_dataset.save_to_disk(partner_mentions_path)
        print(f"num partner_mentions examples: {len(partner_mentions_examples['input'])}")

        # partner mention ind | partner markers, textmention history, dots
        # use ground truth mentions
        num_examples = len(examples["raw_partner_mentions"])
        raw_partner_mentions_examples = {}
        raw_partner_mentions_examples["input"] = [
            f"{text} [MSEP] {lasttext}"
            for (
                text,
                lasttext,
            ) in zip(
                examples["prev_text_mentions"]
                    if not options.last_last_turn
                    else examples["lastlasttext_mentions"],
                examples["partner_markers"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in raw_partner_mentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of raw_partner_mentions input: {max_length_input}")
        print(raw_partner_mentions_examples["input"][np.argmax(input_lens)])

        raw_partner_mentions_examples["label"] = examples["raw_partner_mentions"]

        # add metadata
        raw_partner_mentions_examples["chat_id"] = examples["chat_id"]
        raw_partner_mentions_examples["scenario_id"] = examples["scenario_id"]
        raw_partner_mentions_examples["agent"] = examples["agent"]
        raw_partner_mentions_examples["dots"] = examples["raw_dots"]

        raw_partner_mentions_dataset = Dataset.from_dict(raw_partner_mentions_examples)
        raw_partner_mentions_path = f"hf_datasets/{split}_raw_partner_mentions_{feature_string}.hf"
        print(f"raw_partner_mentions dataset path {raw_partner_mentions_path}")
        raw_partner_mentions_dataset.save_to_disk(raw_partner_mentions_path)
        print(f"num raw_partner_mentions examples: {len(raw_partner_mentions_examples['input'])}")

        # partner mention joint | partner markers, textmention history, dots
        # use ground truth mentions
        num_examples = len(examples["joint_partner_mentions"])
        joint_partner_mentions_examples = {}
        joint_partner_mentions_examples["input"] = [
            f"{text} [MSEP] {lasttext}"
            for (
                text,
                lasttext,
            ) in zip(
                examples["prev_text_mentions"]
                    if not options.last_last_turn
                    else examples["lastlasttext_mentions"],
                examples["partner_markers"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in joint_partner_mentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of joint_partner_mentions input: {max_length_input}")
        print(joint_partner_mentions_examples["input"][np.argmax(input_lens)])

        joint_partner_mentions_examples["label"] = examples["joint_partner_mentions"]
        max_length_output = max([len(x) for x in examples["joint_partner_mentions"]])
        print(f"Max length of joint_partner_mentions output: {max_length_output}")

        # add metadata
        joint_partner_mentions_examples["chat_id"] = examples["chat_id"]
        joint_partner_mentions_examples["scenario_id"] = examples["scenario_id"]
        joint_partner_mentions_examples["agent"] = examples["agent"]
        joint_partner_mentions_examples["dots"] = examples["raw_dots"]

        joint_partner_mentions_dataset = Dataset.from_dict(joint_partner_mentions_examples)
        joint_partner_mentions_path = f"hf_datasets/{split}_joint_partner_mentions_{feature_string}.hf"
        print(f"joint_partner_mentions dataset path {joint_partner_mentions_path}")
        joint_partner_mentions_dataset.save_to_disk(joint_partner_mentions_path)
        print(f"num joint_partner_mentions examples: {len(joint_partner_mentions_examples['input'])}")

        # partner markers | partner text
        # use ground truth mentions
        num_examples = len(examples["partner_markers"])
        partner_markers_examples = {}
        partner_markers_examples["input"] = examples["lasttext"]
        input_lens = [len(tokenizer.tokenize(x)) for x in partner_markers_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of partner_markers input: {max_length_input}")
        print(partner_markers_examples["input"][np.argmax(input_lens)])

        partner_markers_examples["label"] = examples["partner_markers"]
        #partner_markers_examples["label"] = [
            #" ".join([str(1) if x else str(0) for x in markers])
            #for markers in examples["partner_markers"]
        #]
        output_lens = [len(tokenizer.tokenize(x)) for x in partner_markers_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of partner_markers output: {max_length_output}")
        print(partner_markers_examples["label"][np.argmax(output_lens)])

        # add metadata
        partner_markers_examples["chat_id"] = examples["chat_id"]
        partner_markers_examples["scenario_id"] = examples["scenario_id"]
        partner_markers_examples["agent"] = examples["agent"]
        partner_markers_examples["dots"] = examples["raw_dots"]

        partner_markers_dataset = Dataset.from_dict(partner_markers_examples)
        partner_markers_path = f"hf_datasets/{split}_partner_markers_{feature_string}.hf"
        print(f"partner_markers dataset path {partner_markers_path}")
        partner_markers_dataset.save_to_disk(partner_markers_path)
        print(f"num partner_markers examples: {len(partner_markers_examples['input'])}")

        # partner tags | partner text
        # use ground truth mentions
        num_examples = len(examples["partner_tags"])
        partner_tags_examples = {}
        partner_tags_examples["input"] = examples["lasttext"]
        input_lens = [len(tokenizer.tokenize(x)) for x in partner_tags_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of partner_tags input: {max_length_input}")
        print(partner_tags_examples["input"][np.argmax(input_lens)])

        partner_tags_examples["label"] = examples["partner_tags"]

        # add metadata
        partner_tags_examples["chat_id"] = examples["chat_id"]
        partner_tags_examples["scenario_id"] = examples["scenario_id"]
        partner_tags_examples["agent"] = examples["agent"]
        partner_tags_examples["dots"] = examples["raw_dots"]

        partner_tags_dataset = Dataset.from_dict(partner_tags_examples)
        partner_tags_path = f"hf_datasets/{split}_raw_partner_tags_{feature_string}.hf"
        print(f"raw_partner_tags dataset path {partner_tags_path}")
        partner_tags_dataset.save_to_disk(partner_tags_path)
        print(f"num raw_partner_tags examples: {len(partner_tags_examples['input'])}")

        # partner mention | partner aligned markers, textmention history, dots
        # use ground truth mentions
        num_examples = len(examples["partner_amentions"])
        partner_amentions_examples = {}
        partner_amentions_examples["input"] = [
            f"{text} [MSEP] {lasttext}"
            for (
                text,
                lasttext,
            ) in zip(
                examples["prev_text_mentions"]
                    if not options.last_last_turn
                    else examples["lastlasttext_mentions"],
                examples["partner_amarkers"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in partner_amentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of partner_amentions input: {max_length_input}")
        print(partner_amentions_examples["input"][np.argmax(input_lens)])

        partner_amentions_examples["label"] = examples["partner_amentions"]
        output_lens = [len(tokenizer.tokenize(x)) for x in partner_amentions_examples["label"]]
        max_length_output = max(output_lens)
        print(f"Max length of partner_amentions output: {max_length_output}")
        print(partner_amentions_examples["label"][np.argmax(output_lens)])

        # add metadata
        partner_amentions_examples["chat_id"] = examples["chat_id"]
        partner_amentions_examples["scenario_id"] = examples["scenario_id"]
        partner_amentions_examples["agent"] = examples["agent"]
        partner_amentions_examples["dots"] = examples["raw_dots"]

        partner_amentions_dataset = Dataset.from_dict(partner_amentions_examples)
        partner_amentions_path = f"hf_datasets/{split}_partner_amentions_{feature_string}.hf"
        print(f"partner_amentions dataset path {partner_amentions_path}")
        partner_amentions_dataset.save_to_disk(partner_amentions_path)
        print(f"num partner_amentions examples: {len(partner_amentions_examples['input'])}")

        # agent mention joint | agent markers, last partner textmentions, dots
        # use ground truth mentions
        num_examples = len(examples["joint_mentions"])
        joint_mentions_examples = {}
        joint_mentions_examples["input"] = [
            f"{lasttext_mention} [MSEP] {markers}"
            for (
                lasttext_mention,
                markers,
            ) in zip(
                examples["lasttext_mentions"],
                examples["markers"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in joint_mentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of joint_mentions input: {max_length_input}")
        print(joint_mentions_examples["input"][np.argmax(input_lens)])

        joint_mentions_examples["label"] = examples["joint_mentions"]
        max_length_output = max([len(x) for x in examples["joint_mentions"]])
        print(f"Max length of joint_mentions output: {max_length_output}")

        # add metadata
        joint_mentions_examples["chat_id"] = examples["chat_id"]
        joint_mentions_examples["scenario_id"] = examples["scenario_id"]
        joint_mentions_examples["agent"] = examples["agent"]
        joint_mentions_examples["dots"] = examples["raw_dots"]

        joint_mentions_dataset = Dataset.from_dict(joint_mentions_examples)
        joint_mentions_path = f"hf_datasets/{split}_joint_mentions_{feature_string}.hf"
        print(f"joint_mentions dataset path {joint_mentions_path}")
        joint_mentions_dataset.save_to_disk(joint_mentions_path)
        print(f"num joint_mentions examples: {len(joint_mentions_examples['input'])}")


        # agent tags | agent text
        # use ground truth mentions
        num_examples = len(examples["tags"])
        tags_examples = {}
        tags_examples["input"] = examples["outtext"]
        input_lens = [len(tokenizer.tokenize(x)) for x in tags_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of tags input: {max_length_input}")
        print(tags_examples["input"][np.argmax(input_lens)])

        tags_examples["label"] = examples["tags"]

        # add metadata
        tags_examples["chat_id"] = examples["chat_id"]
        tags_examples["scenario_id"] = examples["scenario_id"]
        tags_examples["agent"] = examples["agent"]
        tags_examples["dots"] = examples["raw_dots"]

        tags_dataset = Dataset.from_dict(tags_examples)
        tags_path = f"hf_datasets/{split}_raw_tags_{feature_string}.hf"
        print(f"raw_tags dataset path {tags_path}")
        tags_dataset.save_to_disk(tags_path)
        print(f"num raw_tags examples: {len(tags_examples['input'])}")

        # mention ind | markers, lasttextmention history, dots
        # use ground truth mentions
        num_examples = len(examples["raw_mentions"])
        raw_mentions_examples = {}
        raw_mentions_examples["input"] = [
            f"{lasttext_mention} [MSEP] {markers}"
            for (
                lasttext_mention,
                markers,
            ) in zip(
                examples["lasttext_mentions"],
                examples["markers"],
            )
        ]
        input_lens = [len(tokenizer.tokenize(x)) for x in raw_mentions_examples["input"]]
        max_length_input = max(input_lens)
        print(f"Max length of raw_mentions input: {max_length_input}")
        print(raw_mentions_examples["input"][np.argmax(input_lens)])

        raw_mentions_examples["label"] = examples["raw_mentions"]
        max_length_output = max([len(x) for x in examples["raw_mentions"]])
        print(f"Max length of raw_mentions output: {max_length_output}")

        # add metadata
        raw_mentions_examples["chat_id"] = examples["chat_id"]
        raw_mentions_examples["scenario_id"] = examples["scenario_id"]
        raw_mentions_examples["agent"] = examples["agent"]
        raw_mentions_examples["dots"] = examples["raw_dots"]

        raw_mentions_dataset = Dataset.from_dict(raw_mentions_examples)
        raw_mentions_path = f"hf_datasets/{split}_raw_mentions_{feature_string}.hf"
        print(f"raw_mentions dataset path {raw_mentions_path}")
        raw_mentions_dataset.save_to_disk(raw_mentions_path)
        print(f"num raw_mentions examples: {len(raw_mentions_examples['input'])}")
