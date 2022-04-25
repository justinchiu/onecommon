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

#random.seed(1234)
#np.random.seed(1234)

# Testing streamlit
st.title("OneCommon Planning Visualization")

#json_file = "experiments_nov-22/all_dialogues.json"
json_file = "../../onecommon_human_experiments/dialogues_with_completion_info.json"

board_json_file = "../../onecommon_human_experiments/shared_4_100.json"

with open(json_file, "r") as f:
    dialogues = json.load(f)

with open(board_json_file, "r") as f:
    boards = json.load(f)
    # convert boards to map
    boards = {
        board["uuid"]: board
        for board in boards
    }

model_types = set([x["agent_types"]["1"] for x in dialogues])

def chop_up(xs):
    stuff = {}
    for ty in model_types:
        stuff[ty] = list(filter(
            lambda x: x["opponent_type"] == ty,
            xs,
        ))
    return stuff

def get_complete(xs):
    return list(filter(lambda x: x["num_players_selected"] == 2, xs))

def get_selected(xs):
    # num_players_selected is inaccurate, final selection may be dropped
    return list(filter(
        lambda x: x["events"][-2]["action"] == "select" and x["events"][-1]["action"] == "select",
        xs))

def get_success(xs):
    return list(filter(lambda x: x["outcome"]["reward"] == 1 and x["num_players_selected"] == 2, xs))

def apply(dxs, fn):
    d = {}
    for ty, xs in dxs.items():
        d[ty] = fn(xs)
    return d

def get_rates(dxs):
    rates = apply(
        dxs,
        lambda xs: len(get_success(xs)) / len(get_complete(xs)),
    )
    completed = apply(
        dxs,
        lambda xs: len(get_complete(xs)),
    )
    return rates, completed

# heuristics
def min_len(L, xs):
    return [x for x in xs if len(x["dialogue"]) > L]

def dialogue_has_words(x, words):
    has_any = False
    for _, utt in x["dialogue"]:
        utt = utt.lower().split()
        for word in words:
            has_any |= word in utt
    return has_any

def has_spatial_word(xs):
    words = [
        "left",
        "right",
        "above",
        "near",
        "close",
        "alone",
        "far",
        "top",
        "below",
        "bottom",
        "dark",
        "light",
        "black",
        "grey",
        "gray",
    ]
    return [x for x in xs if dialogue_has_words(x, words)]

def has_utt_len(x, L):
    has_long0 = False
    has_long1 = False
    for agent, utt in x["dialogue"]:
        utt = utt.lower().split()
        if agent == 0:
            has_long0 |= len(utt) >= L
        else:
            has_long1 |= len(utt) >= L
    return has_long0 and has_long1

def min_utt_len(L, xs):
    return [x for x in xs if has_utt_len(x, L)]

def apply_mul(dxs, fns):
    d = {}
    for ty, xs in dxs.items():
        acc = xs
        for fn in fns:
            acc = fn(acc)
        d[ty] = acc
    return d

# get success rates?
dialogues_by_ty = chop_up(get_complete(dialogues))
finished_dialogues_by_ty = chop_up(get_selected(dialogues))

def visualize_board(left_dots, right_dots, select0, select1, intersect):
    shift = 430

    left_dots_html = map(lambda x: x.html(), left_dots)
    right_dots_html = map(lambda x: x.html(shift), right_dots)

    intersect_dots = map(lambda x: x.intersect_html(), intersect)

    select_left = list(filter(lambda x: int(x.id) == select0, left_dots))[0]
    select_right = list(filter(lambda x: int(x.id) == select1, right_dots))[0]


    nl = "\n"
    html = f"""
    <svg width="860" height="430">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(left_dots_html)}
    {nl.join(intersect_dots)}
    {select_left.select_html()}
    <circle cx="645" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(right_dots_html)}
    {select_right.select_html(shift=shift)}
    </svg>
    """
    components.html(html, height=430, width=860)

def visualize_dialogue(dialogue):
    st.table(dialogue)

"""
train_json_file = "../../aaai2020/experiments/data/onecommon/final_transcripts.json"
with open(train_json_file, "r") as f:
    train_dialogues = json.load(f)
id2dialogue = {x["scenario_uuid"]: x for x in train_dialogues}
def process_dialogue(dialogue_dict):
    scenario_id = dialogue_dict["scenario_id"]
    dialogue = dialogue_dict["dialogue"]
    agent_types = dialogue_dict["agent_types"]

    reward = dialogue_dict["outcome"]["reward"]
    event0 = dialogue_dict["events"][-2]
    event1 = dialogue_dict["events"][-1]
    select0 = int((event0["data"] if event0["agent"] == 0 else event1["data"]).replace("\"", ""))
    select1 = int((event0["data"] if event1["agent"] == 0 else event1["data"]).replace("\"", ""))


    board = boards[scenario_id]
    st.write(f"Chat scenario id: {scenario_id}")
    st.write("SUCCESS" if reward else "FAILURE")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect = [x for x in b0 for y in b1 if x.id == y.id]

    visualize_board(b0, b1, select0, select1, intersect)

    st.write(f"Agent 0: {agent_types['0']} || 1: {agent_types['1']}")
    visualize_dialogue(dialogue)
"""

def visualize_board(left_dots, right_dots, mentions, intersect, beliefs=None):
    shift = 430

    left_dots_html = (map(lambda x: x.html(), left_dots)
        if beliefs is None
        else map(lambda x: x[0].html(value=x[1]), zip(left_dots, beliefs))
    )
    right_dots_html = map(lambda x: x.html(shift), right_dots)

    mentions_html = map(lambda x: x.select_html(), mentions)
    intersect_dots = map(lambda x: x.intersect_html(), intersect)

    nl = "\n"
    html = f"""
    <svg width="860" height="430">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(left_dots_html)}
    {nl.join(intersect_dots)}
    {nl.join(mentions_html)}
    <circle cx="645" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(right_dots_html)}
    </svg>
    """
    components.html(html, height=430, width=860)

def visualize_dialogue(dialogue):
    st.table(dialogue)

def process_dialogue(scenario_id, dialogue):
    board = boards[scenario_id]
    st.write(f"Chat scenario id: {scenario_id}")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect = [x for x in b0 for y in b1 if x.id == y.id]

    #turn = st.radio("Turn number", np.arange(len(dialogue)))
    turn = st.number_input("Turn number", 0, len(dialogue)-1)

    st.header("Dialogue so far")
    for t in range(turn):
        st.write(f"u: {dialogue[t]['utterance_language']} || r: {dialogue[t]['response_language']}")

    turn = dialogue[turn]

    display = st.radio("Display", ("prior", "plan", "posterior"))

    if display == "prior":
        st.header("Prior next mentions")
        prior_mentions = turn["prior_mentions"]
        prior_idx = st.radio("Prior number", np.arange(len(prior_mentions)))
        mention = [x for i,x in enumerate(b0) if prior_mentions[prior_idx][i] == 1]
        visualize_board(b0, b1, mention, intersect)
    elif display == "plan":
        st.header("Plan next mentions")
        plan_mentions = turn["plan_mentions"]
        plan_idx = st.radio("Plan number", np.arange(len(plan_mentions)))
        mention = [x for i,x in enumerate(b0) if plan_mentions[plan_idx][i] == 1]
        visualize_board(b0, b1, mention, intersect)
    elif display == "posterior":
        st.header("Input and belief posterior")    
        st.write(f"Input utterance: {turn['utterance_language']} || response: {turn['response_language']}")
        mention = [x for i,x in enumerate(b0) if turn["utterance"][i] == 1]
        visualize_board(b0, b1, mention, intersect, beliefs=turn["marginal_belief"])



dir = "../../aaai2020/experiments/analysis_log"
scenario_id = "S_pGlR0nKz9pQ4ZWsw"
filepath = (Path(dir) / scenario_id).with_suffix(".json")

with filepath.open() as f:
    dialogue = json.load(f)

process_dialogue(scenario_id, dialogue)
