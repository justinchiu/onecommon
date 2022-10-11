from typing import NamedTuple

from pathlib import Path
import numpy as np
from jinja2 import Template
from rich.progress import track
import json

import torch

from belief import process_ctx

from datasets import Dataset, load_dataset

from domain import get_domain

from corpora.data import Dictionary, get_tag
from corpora.reference_sentence import ReferenceSentenceCorpus

import template

from hfutils import (
    HfDataOptions, Property, DescriptionFormat,
    construct_feature_string, get_bart_tokenizer,
)

fields = (
    "dots",
    "plan_specific_dots",
    "text",
    "plan",
    "mentions",
    "confirmation",
    "selection_leaning",
    "selection",
    "outtext",
    "text_mentions",
    "outtext_mentions",
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
][7]

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

def splice_sentence_mentions(sent, refs):
    refs = np.array(refs).reshape(-1, 10)
    sent = [x for x in sent]
    for idx, ref in reversed(list(zip(refs[:,0], refs[:,3:]))):
        sent.insert(idx, "<bom>")
        num_added = 1
        for i, val in enumerate(ref):
            if val:
                sent.insert(idx + num_added, f"dot{i+1}")
                num_added += 1
        sent.insert(idx+num_added, "<eom>")
    return sent

"""
REFERENT LAYOUT: [
    beginning idx,
    ending idx,
    utterance end idx,
    *binary indicators for occurrence, eg 7 dots 0 1 1 0 0 0 0 => dots 1 and 2
]
"""
def _split_referents(words, referent_idxs):
    stops = ["YOU:", "THEM:"]
    sents_mentions = []
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
                sents_mentions.append(splice_sentence_mentions(current, current_refs))
            current = []
            current_refs = []
        current.append(w)
    if len(current) > 0:
        while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
            current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
            ref_ix += 1
        sents.append(current)
        all_refs.append(current_refs)
        sents_mentions.append(splice_sentence_mentions(current, current_refs))
    assert ref_ix == len(split_ref_indices)
    assert sum(len(refs) for refs in all_refs) == len(referent_idxs)
    assert len(all_refs) == len(sents)
    return sents, all_refs, sents_mentions

class Conversation(NamedTuple):
    dots: list[float]
    sents: list[list[str]]
    refs: list[list[int]]
    partner_refs: list[list[int]]
    partner_refs_our_view: list[list[int]]

    scenario_id: str
    chat_id: str

    real_ids: list[str]
    partner_real_ids: list[str]

    # added, splices in mentions into sentences
    sentrefs: list[list[str]]
    partner_sentrefs: list[list[str]]
    partner_sentrefs_our_view: list[list[str]]
    all_sentrefs: list[list[str]]


def get_conversations(split):
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

            sents, all_refs, sentrefs = _split_referents(words, referent_idxs)
            sents_, all_partner_refs, partner_sentrefs = _split_referents(words, partner_referent_idxs)
            assert sents == sents_
            sents_, all_partner_refs_our_view, partner_sentrefs_our_view = _split_referents(words, partner_referent_our_view_idxs)
            assert sents == sents_

            # merge YOU: and THEM:, all in our view
            all_sentrefs = [
                x if x[0] == "YOU:" else y
                for x,y in zip(sentrefs, partner_sentrefs_our_view)
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

                real_ids = real_ids,
                partner_real_ids = partner_real_ids,

                sentrefs = sentrefs,
                partner_sentrefs = partner_sentrefs,
                partner_sentrefs_our_view = partner_sentrefs_our_view,
                all_sentrefs = all_sentrefs,
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
    vert_comp = "above" if y1 > y2 else "below"
    hor_comp = "right" if x1 > x2 else "left"
    size_comp = "bigger" if s1 > s2 else "smaller"
    col_comp = "darker" if c1 > c2 else "lighter"

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

"""
Process conversations into examples.
Each conversation will be turned into num_turns examples (or num_turns / 2).
Each example will look like:
    * All previous turns: each turn is a list of tokens (strings)
    * All previous refs, partner refs, partner_refs_our_view
    * scenario, chat, and ids
Eg conversation prefixes
"""

def describe_dots(
    dots,
    use_short_describe = True,
    use_pairwise_features = True,
    use_unordered_pairwise = True,
    use_short_pairwise = True,
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

    return description


def describe_plan_specific_dots(
    dots,
    plan,
    use_unordered_pairwise = True,
    close_dots = None,
    use_short_pairwise = True,
    format = DescriptionFormat.SrcRelTgt,
):
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

        return description
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
        return description
    else:
        raise ValueError(f"Invalid format: {format.name}")

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
    #if any([word not in word_dict.word2idx for sent in sents for word in sent]):
        #import pdb; pdb.set_trace()
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

            if options.max_plan_size > 0 and raw_plan.sum() > options.max_plan_size:
                num_skipped += 1
                #print(" ".join(conversation.sents[turn]))
                #import pdb; pdb.set_trace()
                continue

            is_you = conversation.sents[turn][0] == "YOU:"
            if is_you:
                # textify all dot properties

                # mention = (start idx, end idx, utterance end idx, *binary ind for 7 dots)
                examples["plan"].append(describe_plan(raw_plan))
                examples["mentions"].append(
                    " [SEP] ".join([describe_plan(m) for m in raw_mentions])
                )

                # linearized dot representation
                examples["dots"].append(describe_dots(
                    conversation.dots,
                    use_short_describe = options.short_describe,
                    use_pairwise_features = True,
                    use_unordered_pairwise = options.unordered_rel,
                    use_short_pairwise = options.short_rel,
                ))

                # plan-specific dot representations
                examples["plan_specific_dots"].append(
                    describe_plan_specific_dots(
                        conversation.dots,
                        raw_plan,
                        options.unordered_rel,
                        close_dots = None,
                        use_short_pairwise = options.short_rel,
                        format = options.format,
                    ),
                )

                # concatenate all text
                examples["text"].append(
                    " ".join([x for xs in conversation.sents[:turn] for x in xs])
                )

                sent = " ".join(conversation.sents[turn])
                examples["outtext"].append(sent)

                # confirmation
                confirmation_prediction = confirmations[conversation.scenario_id][str(turn)]
                # 0: None, 1: Confirm, 2: Disconfirm
                confirm_map = ["none", "yes", "no"]
                examples["confirmation"].append(
                    f"confirmation: {confirm_map[confirmation_prediction]}"
                )

                # selection-leaning
                selection_like_words = set(["pick", "choose", "select"])
                has_select = any([
                    x in selection_like_words for x in conversation.sents[turn]
                ])
                examples["selection_leaning"].append(
                    "should we select? yes"
                    if has_select
                    else "should we select? no"
                )

                # selection
                examples["selection"].append(
                    "<selection>"
                    if turn == num_turns - 1
                    else "selection: not yet"
                )

                # sentrefs
                examples["outtext_mentions"].append(
                    " ".join(conversation.sentrefs[turn])
                )
                examples["text_mentions"].append(
                    " ".join([x for xs in conversation.all_sentrefs[:turn] for x in xs])
                )

            #for field in examples.keys():
                #print(field, examples[field][-1])
            #import pdb; pdb.set_trace()


    # Check number of examples for each field
    for key1 in fields:
        for key2 in fields:
            assert len(examples[key1]) == len(examples[key2])
    print(f"Number of examples skipped due to plan size: {num_skipped}")
    return examples

if __name__ == "__main__":
    redo_examples = True
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
    )

    print(Conversation._fields)

    response_pretrained_path = "models/save_pretrained"
    confirmation_tokenizer = AutoTokenizer.from_pretrained(response_pretrained_path)
    confirmation_predictor = AutoModelForSequenceClassification.from_pretrained(
        response_pretrained_path)

    for split in splits:
        conversations = get_conversations(split)

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


        dot_descs = (
            examples["plan_specific_dots"]
            if options.plan_specific_description
            else examples["dots"]
        )

        # mention gen
        num_examples = len(examples["mentions"])
        mention_examples = {}
        mention_examples["input"] = [
            f"{dots} [MSEP] {text} [MSEP] {plan}"
            for dots, text, plan in zip(dot_descs, examples["text"], examples["plan"])
        ]
        mention_examples["label"] = examples["mentions"]
        mention_dataset = Dataset.from_dict(mention_examples)
        mention_path = f"hf_datasets/{split}_mentions_given_text_plan_{feature_string}.hf"
        print(f"Mention dataset path {mention_path}")
        mention_dataset.save_to_disk(mention_path)

        # plan gen
        num_examples = len(examples["plan"])
        plan_examples = {}
        #plan_examples["input"] = examples["text"]
        plan_examples["input"] = [
            f"{dots} [MSEP] {text}"
            for dots, text in zip(examples["dots"], examples["text"])
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
        if not options.dialog_history:
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
        textmention_mention_dataset = Dataset.from_dict(textmention_mention_examples)
        textmention_mention_path = f"hf_datasets/{split}_textmention_mention_given_plan_{feature_string}.hf"
        print(f"Textmention,mention dataset path {textmention_mention_path}")
        textmention_mention_dataset.save_to_disk(textmention_mention_path)
        print(f"num textmention,mention examples: {len(textmention_mention_examples['input'])}")

