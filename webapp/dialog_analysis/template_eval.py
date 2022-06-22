import sys

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from functools import partial

import random
import json
from functools import partial
import numpy as np
from scipy.special import logsumexp as lse

from dot import Dot

sys.path.append("/home/justinchiu/research/onecommon/aaai2020/experiments")

from utils import ContextGenerator
import template
import template_rec as tr
import belief

#random.seed(1234)
#np.random.seed(1234)

# Testing streamlit
st.title("OneCommon Template Visualization")

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

def visualize_single_board(dots):
    shift = 430

    dots_html = map(lambda x: x.html(), dots)

    nl = "\n"
    html = f"""
    <svg width="450" height="500" overflow="visible">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(dots_html)}
    </svg>
    """
    components.html(html, height=500, width=450)

def visualize_dialogue(dialogue):
    st.table(dialogue)

def visualize_beliefs(dots, configs, belief):
    st.write("Configs (Belief)")
    for row, b in zip(configs, belief):
        st.write(f'{" ".join([str(x.id) for on, x in zip(row, dots) if on])} ({b:.4f})')

def process_dialogue(scenario_id, dialogue):
    board = boards[scenario_id]
    st.write(f"Chat scenario id: {scenario_id}")
    st.write(f"Num turns: {len(dialogue)}")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]

    #turn = st.radio("Turn number", np.arange(len(dialogue)))
    
    turn = st.number_input("Turn number", 0, len(dialogue)-1)
    #turn = 2

    st.header("Dialogue so far")
    for t in range(turn):
        st.write(f"Agent {dialogue[t]['writer_id']}: {dialogue[t]['utterance_language']}")

    turn = dialogue[turn]

    """
    display = st.radio("Display", (
        "prior",
        "plan", "plan2", "plan3",
        "posterior", "posterior2", "posterior3",
        "priorbeam", "planbeam",
    ))
    """
    # FOR TEMPLATE EVAL
    display = "planbeam"

    #st.write(f"### BEAM SEARCH for agent {turn['writer_id']}")

    mentions = turn["plan_beam_seed"]
    resolved = turn["prior_partner_ref" if display == "priorbeam" else "plan3_partner_ref"]
    resolved = np.array(resolved).any(0)[0]
    if display == "priorbeam":
        mentions = np.array(turn["prior_plan"]).any(0)[0]
    if turn["writer_id"] == 0:
        mention0 = [x for i,x in enumerate(b0) if mentions[i] == 1]
        mention1 = [x for i,x in enumerate(b1) if resolved[i] == 1]

        m = mention0
        board_agent = b0
        board_partner = b1
    elif turn["writer_id"] == 1:
        mention0 = [x for i,x in enumerate(b0) if resolved[i] == 1]
        mention1 = [x for i,x in enumerate(b1) if mentions[i] == 1]

        m = mention1
        board_agent = b1
        board_partner = b0

    st.write(f"### Partner perspective")
    visualize_single_board(board_partner)


    st.write("### template utterance")
    ctx_struct = ctxs[scenario_id]
    ctx = ctx_struct[1 + turn["writer_id"]]
    ctx_np = np.array(ctx, dtype=float).reshape(7, 4)
    from belief import OrBelief
    belief = OrBelief(7, ctx_np)
    n, sc, xy = belief.get_feats(np.array(mentions))
    #print(ctx_struct[3 + turn["writer_id"]])
    #print(mentions)
    #print(xy)
    #words = template.render(n, sc, xy)
    words = tr.render(n, sc, xy)
    st.write(words)
    #import pdb; pdb.set_trace()

    st.write("Dot set (out of order)")
    st.write(m)

    st.write(f"### Agent perspective")
    visualize_single_board(board_agent)
    st.write("### template utterance repeat")
    st.write(words)

    st.write("Size and color buckets (row = dot)")
    st.write(sc)

    st.write("### chosen utterance")
    st.write(turn["prior_mentions_language"
        if display == "priorbeam" else "plan3_mentions_language"])

split = "train"
split = "valid_1"
analysis_path = Path("../../aaai2020/experiments/analysis_log") / split
scenarios = [f.stem for f in analysis_path.iterdir() if f.is_file()]

idx = st.number_input("Scenario number", 0, len(scenarios))
#idx = 1
scenario_id = scenarios[idx]

filepath = (analysis_path/ scenario_id).with_suffix(".json")

with filepath.open() as f:
    dialogue = json.load(f)
    process_dialogue(scenario_id, dialogue)
