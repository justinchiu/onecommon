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

    if display == "prior":
        st.header("Prior next mentions")
        prior_mentions = turn["prior_mentions"]
        if prior_mentions is not None:
            prior_idx = st.radio("Prior number", np.arange(len(prior_mentions)))
            if turn["writer_id"] == 0:
                mention0 = [x for i,x in enumerate(b0) if prior_mentions[prior_idx][i] == 1]
                mention1 = None
            elif turn["writer_id"] == 1:
                mention1 = [x for i,x in enumerate(b1) if prior_mentions[prior_idx][i] == 1]
                mention0 = None
            visualize_board(b0, b1, mention0, mention1, intersect0, intersect1)
        else:
            st.write("No prior next mentions")
    elif display == "plan" or display == "plan2" or display == "plan3":
        st.header("Plan next mentions")
        if display == "plan":
            plan_mentions = turn["plan_mentions"]
        elif display == "plan2":
            plan_mentions = turn["plan2_mentions"]
        elif display == "plan3":
            plan_mentions = turn["plan3_mentions"]
        if plan_mentions is not None:
            plan_idx = st.radio("Plan number", np.arange(len(plan_mentions)))
            if turn["writer_id"] == 0:
                mention0 = [x for i,x in enumerate(b0) if plan_mentions[plan_idx][i] == 1]
                mention1 = None
            elif turn["writer_id"] == 1:
                mention1 = [x for i,x in enumerate(b1) if plan_mentions[plan_idx][i] == 1]
                mention0 = None
            visualize_board(b0, b1, mention0, mention1, intersect0, intersect1)
        else:
            st.write("No plan mentions")
    elif display == "posterior" or display == "posterior2" or display == "posterior3":
        st.header(f"Input and belief posterior for agent {turn['reader_id']}")    
        st.write(f"Utterance from agent {turn['writer_id']}: {turn['utterance_language']}")
        if turn["reader_id"] == 0:
            mention0 = (
                [x for i,x in enumerate(b0) if turn["response_utt"][i] == 1]
                if turn["response_utt"] is not None else None
            )
            mention1 = (
                [x for i,x in enumerate(b1) if turn["utterance"][i] == 1]
                if turn["utterance"] is not None else None
            )
        elif turn["reader_id"] == 1:
            mention0 = (
                [x for i,x in enumerate(b0) if turn["utterance"][i] == 1]
                if turn["utterance"] is not None else None
            )
            mention1 = (
                [x for i,x in enumerate(b1) if turn["response_utt"][i] == 1]
                if turn["response_utt"] is not None else None
            )
        if display == "posterior":
            marginals = turn["marginal_belief"]
        elif display == "posterior2":
            marginals = turn["marginal_belief2"]
        elif display == "posterior3":
            marginals = turn["marginal_belief3"]
        visualize_board(
            b0, b1, mention0, mention1, intersect0, intersect1,
            left_beliefs=marginals if turn["reader_id"] == 0 else None,
            right_beliefs=marginals if turn["reader_id"] == 1 else None,
        )
        if display == "posterior":
            belief = turn["belief"]
            configs = turn["configs"]
        elif display == "posterior2":
            belief = turn["belief2"]
            configs = turn["configs2"]
        elif display == "posterior3":
            belief = turn["belief3"]
            configs = turn["configs3"]
        logits = ", ".join([f"{x:.2f}" for x in turn["response_logits"]])
        st.write(f"Response: {turn['response_label']}, logits: [{logits}]")
        visualize_beliefs(
            b0 if turn["reader_id"] == 0 else b1,
            configs,
            belief,
        )
    elif display == "priorbeam" or display == "planbeam":
        st.write(f"### BEAM SEARCH for agent {turn['writer_id']}")
        mentions = turn["plan_beam_seed"]
        resolved = turn["prior_partner_ref" if display == "priorbeam" else "plan3_partner_ref"]
        resolved = np.array(resolved).any(0)[0]
        if display == "priorbeam":
            mentions = np.array(turn["prior_plan"]).any(0)[0]
        if turn["writer_id"] == 0:
            mention0 = [x for i,x in enumerate(b0) if mentions[i] == 1]
            mention1 = [x for i,x in enumerate(b1) if resolved[i] == 1]
        elif turn["writer_id"] == 1:
            mention0 = [x for i,x in enumerate(b0) if resolved[i] == 1]
            mention1 = [x for i,x in enumerate(b1) if mentions[i] == 1]
        visualize_board(b0, b1, mention0, mention1, intersect0, intersect1)


        st.write("### template utterance")
        ctx_struct = ctxs[scenario_id]
        ctx = ctx_struct[1 + turn["writer_id"]]
        ctx_np = np.array(ctx, dtype=float).reshape(7, 4)
        from belief import OrBelief
        belief = OrBelief(7, ctx_np)
        feats = belief.get_feats(np.array(mentions))
        ids = np.array(mentions).nonzero()[0]
        #print(ctx_struct[3 + turn["writer_id"]])
        #print(mentions)
        #print(xy)
        #words = template.render(n, sc, xy)
        words = tr.render(*feats, ids, confirm=None)
        st.write(words)
        #import pdb; pdb.set_trace()


        st.write("### chosen utterance")
        st.write(turn["prior_mentions_language"
            if display == "priorbeam" else "plan3_mentions_language"])
        st.write("### beam output")
        sents = turn["prior_beam_sents" if display == "priorbeam" else "plan_beam_sents"]
        ref_res = turn["prior_beam_ref_res" if display == "priorbeam" else "plan_beam_ref_res"]
        lm = turn["prior_beam_lm" if display == "priorbeam" else "plan_beam_lm"]
        for s,r,l in sorted(zip(sents, ref_res, lm), reverse=True, key=lambda x: x[1]):
            st.write(s)
            st.write(f"Ref res {r:.2f} LM {l:.2f}")

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
