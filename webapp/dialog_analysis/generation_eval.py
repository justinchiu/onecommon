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

    raw_mentions = input.split("[MSEP]")[-1].split("[SEP]")
    dots = set([
        int(d.strip()[-1])-1
        for m in raw_mentions
        for d in m.split(",")
        if "dot" in d
    ])
    mentions = [b[d] for d in dots]
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
    # plan-specific
    "hf-generations-textmention_given_mention"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd__c_sl_s_mps25__ma_-l1e-05-b4/"
        "checkpoint-14000.gen.json",
    # mention-specific
    "hf-generations-textmention_given_mention"
        "_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_mps25__ma_-l1e-05-b4/"
        "checkpoint-14000.gen.json",
]

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--file", type=int, default=0)
args = parser.parse_args()

generation_file = Path("../../aaai2020/experiments") / generation_files[args.file]

with generation_file.open("r") as f:
    ids_inputs_labels_gens = json.load(f)
    size_to_examples = defaultdict(list)
    for x in ids_inputs_labels_gens:
        input = x[3]
        raw_mentions = input.split("[MSEP]")[-1].split("[SEP]")
        dots = set([
            int(d.strip()[-1])-1
            for m in raw_mentions
            for d in m.split(",")
            if "dot" in d
        ])
        size_to_examples[len(dots)].append(x)

    plan_size = st.number_input("Plan size", 2, 5)

    examples = size_to_examples[plan_size]

    idx = st.number_input("Scenario number", 0, len(examples))
    process_dialogue(examples[idx])
