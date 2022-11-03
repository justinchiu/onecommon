import sys

from pathlib import Path
from collections import defaultdict

import streamlit as st
import streamlit.components.v1 as components

import random
import json
import numpy as np

from dot import Dot

import os
sys.path.append(str((Path.cwd() / "../../aaai2020/experiments").resolve()))

from utils import ContextGenerator

#random.seed(1234)
#np.random.seed(1234)

def extract_mentions(text):
    tokens = text.split()
    idx = 0
    mentions = [] # List[starting idx, dots, ...]
    while idx < len(tokens):
        # check for bom
        if tokens[idx] == "<bom>":
            dots = [idx]
            idx += 1
            for end_idx in range(idx, len(tokens)):
                if tokens[end_idx] == "<eom>":
                    idx = end_idx
                    break
                dots.append(tokens[end_idx])
            mentions.append(dots)
        idx += 1
    return mentions

# Testing streamlit
st.title("OneCommon Planning Visualization")

json_file = "../../aaai2020/experiments/data/onecommon/final_transcripts.json"
with open(json_file, "r") as f:
    dialogues = json.load(f)
dialogues = {
    d["scenario"]["uuid"]: d for d in dialogues
}
with open('../../aaai2020/experiments/data/onecommon/static/scenarios.json', "r") as f:
    scenario_list = json.load(f)
boards = {
    scenario['uuid']: scenario
    for scenario in scenario_list
}

ctx_file = '../../aaai2020/experiments/data/onecommon/static/valid_context_1.txt'
ctx_gen = ContextGenerator(ctx_file)
ctxs = {
    ctx[0][0]: ctx for ctx in ctx_gen.ctxs
}

def visualize_board(
    left_dots, right_dots,
    left_mentions, right_mentions,
    left_intersect, right_intersect,
    left_beliefs=None, right_beliefs=None,
):
    shift = 430

    left_dots_html = (map(lambda x: x.html(), left_dots)
        if left_beliefs is None
        else map(lambda x: x[0].html(value=x[1]), zip(left_dots, left_beliefs))
    )
    right_dots_html = (map(lambda x: x.html(shift), right_dots)
        if right_beliefs is None
        else map(lambda x: x[0].html(shift, value=x[1]), zip(right_dots, right_beliefs))
    )

    if left_mentions is not None:
        left_mentions_html = map(lambda x: x.select_html(), left_mentions)
    if right_mentions is not None:
        right_mentions_html = map(lambda x: x.select_html(shift), right_mentions)
    left_intersect_dots = map(lambda x: x.intersect_html(), left_intersect)
    right_intersect_dots = map(lambda x: x.intersect_html(shift), right_intersect)

    nl = "\n"
    html = f"""
    <svg width="860" height="430">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(left_dots_html)}
    {nl.join(left_intersect_dots)}
    {nl.join(left_mentions_html) if left_mentions is not None else ""}
    <circle cx="645" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(right_dots_html)}
    {nl.join(right_intersect_dots)}
    {nl.join(right_mentions_html) if right_mentions is not None else ""}
    </svg>
    """
    components.html(html, height=430, width=860)
    #components.html(html, height=64, width=128)


def process_dialogue(ids_inputs_labels_gens):
    # i know input is a special function in python. it should not be.
    chat_id, scenario_id, agent, input, label, gens = ids_inputs_labels_gens

    board = boards[scenario_id]
    st.write(f"Chat id: {chat_id}")
    st.write(f"Chat scenario id: {scenario_id}")

    st.write(f"### Agent: {agent}")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]

    b = b0 if agent == 0 else b1

    label_mentions = extract_mentions(label)
    label_dots = set([x for xs in label_mentions for x in xs[1:]])

    gen_mentions = extract_mentions(gens[0])
    gen_dots = set([x for xs in gen_mentions for x in xs[1:]])

    #mentions = [b[d] for d in dots]
    mentions = [b[int(d[-1]) - 1] for d in label_dots]
    m0 = mentions if agent == 0 else None
    m1 = None     if agent == 0 else mentions

    visualize_board(b0, b1, m0, m1, intersect0, intersect1)

    st.write(f"dot ids: {[d.id for d in b]}")

    #st.write(" ".join())
    st.write("### Input")
    st.write(input)
    st.write("### Target")
    st.write(label)
    st.write("### Generations")
    for gen in gens:
        st.write(gen)


generation_files = [
    # 0: reference resolution
    "hf-generations-lasttext_mentions"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps05_dh__ma__rd-l1e-05-b4"
        "-erelation-pa-ri_aj_a-een/"
        "checkpoint-31000.gen.json",
    "hf-generations-lasttext_mentions"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps05_dh__ma__rd-l1e-05-b4"
        "-erelation-pa-ri_pi_aj_pj_a-een/"
        "checkpoint-31000.gen.json",
    "hf-generations-lasttext_mentions"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps05_dh__ma__rd-l1e-05-b4"
        "-erelation-pa-ri_pi_aj_pj_a-eey/"
        "checkpoint-31000.gen.json",
    # 3: ref marker
    "hf-generations-partner_tags"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps05_dh___ma__rd-l1e-05-b4"
        "-erelation-pa-ri_aj_a-een/"
        "checkpoint-31000.gen.json",
    # 4: ref mention
    "hf-generations-partner_mentions"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps05_dh___ma__rd-l1e-05-b4"
        "-erelation-pa-ri_pi_aj_pj_a-een/"
        "checkpoint-31000.gen.json",
]

from argparse import ArgumentParser
parser = ArgumentParser()
#parser.add_argument("--file", type=int, default=0)
parser.add_argument("--file", type=int, default=4)
args = parser.parse_args()

genfile = generation_files[args.file]
generation_file = Path("../../aaai2020/experiments") / genfile

AR_MENTIONS = genfile.split("-")[2][:16] == "partner_mentions"

def split_mentions(xs):
    if xs == "none":
        return []

    split_xs = xs.split("[SEP]")
    mentions = [x.strip().split() for x in split_xs]
    return [[m for m in ms if m != "none"] for ms in mentions]

def evaluate_index_mentions(label_mentions, gen_mentions):
    num_correct, num_examples = 0, 0
    for i, iys in enumerate(label_mentions):
        if len(iys) > 0:
            num_examples += 1
            mention_idx = iys[0]
            ys = set(iys[1:])
            if i < len(gen_mentions):
                xs = set(gen_mentions[i][1:])
                if xs == ys:
                    num_correct += 1
    return num_correct, num_examples

def evaluate_mentions(label_mentions, gen_mentions):
    num_correct, num_examples = 0, 0
    for i, ys in enumerate(label_mentions):
        num_examples += 1
        ys = set(ys)
        if i < len(gen_mentions):
            xs = set(gen_mentions[i])
            if xs == ys:
                num_correct += 1
    return num_correct, num_examples

extract_mentions = extract_mentions if not AR_MENTIONS else split_mentions
evaluate_mentions = evaluate_index_mentions if not AR_MENTIONS else evaluate_mentions

with generation_file.open("r") as f:
    ids_inputs_labels_gens = json.load(f)
    size_to_examples = defaultdict(list)
    num_correct, num_examples = 0, 0
    for i, (cid, sid, agent, input, label, gens) in enumerate(ids_inputs_labels_gens):
        label_mentions = extract_mentions(label)
        gen_mentions = extract_mentions(gens[0])
        dots = (
            set([x for xs in label_mentions for x in xs[1:]])
            if not AR_MENTIONS
            else dots = set([x for xs in label_mentions for x in xs])
        )
        #print(dots)
        #print(label_mentions)
        nc, ne = evaluate_mentions(label_mentions, gen_mentions)
        num_correct += nc
        num_examples += ne

        """
        print(input)
        print(label)
        print(label_mentions)
        print(gens[0])
        print(gen_mentions)
        print(nc, ne)
        """

        size_to_examples[len(dots)].append(ids_inputs_labels_gens[i])
    print(f"{num_correct} / {num_examples}")

    plan_size = st.number_input("Plan size", 2, 5)

    examples = size_to_examples[plan_size]

    idx = st.number_input("Scenario number", 0, len(examples))
    process_dialogue(examples[idx])
