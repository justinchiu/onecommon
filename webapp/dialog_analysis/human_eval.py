import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path

from functools import partial

import random
import json
from functools import partial
import numpy as np

from dot import Dot, visualize_board

random.seed(1234)
np.random.seed(1234)

# Testing streamlit
st.title("OneCommon Train Data Visualizations")

#json_file = "experiments_nov-22/all_dialogues.json"
json_file = "../pomdp_experiments/all_chats.json"

with open(json_file, "r") as f:
    dialogues = json.load(f)

def get_dialogue(x):
    return [y["data"] for y in x["events"] if y["action"] == "message"]

def get_selected(xs):
    # num_players_selected is inaccurate, final selection may be dropped
    return list(filter(
        lambda x: x["events"][-2]["action"] == "select" and x["events"][-1]["action"] == "select",
        xs))

def get_success(xs):
    return list(filter(lambda x: x["outcome"]["reward"] == 1, xs))

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

#finished_dialogues = get_selected(dialogues)
finished_dialogues = dialogues

def visualize_dialogue(dialogue):
    st.write("Agent id (0: left, 1: right), words")
    st.table(dialogue)

def process_dialogue(dialogue_dict):
    id = dialogue_dict["chat_id"]
    scenario_id = dialogue_dict["scenario_id"]
    dialogue = dialogue_dict["dialogue"]
    scenario_id = dialogue_dict["scenario_id"]
    board = boards[scenario_id]
    agent_types = dialogue_dict["agent_types"]

    st.write(f"Chat id: {id}")
    st.write(f"Chat scenario id: {scenario_id}")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]

    events = dialogue_dict["events"]

    reward = dialogue_dict["outcome"]["reward"]
    event0s = [x for x in dialogue_dict["events"] if x["agent"] == 0 and x["action"] == "select"]
    event1s = [x for x in dialogue_dict["events"] if x["agent"] == 1 and x["action"] == "select"]
    mentions0, mentions1 = None,None
    if event0s:
        event0 = event0s[-1]
        select_id0 = int((event0["data"] if event0["agent"] == 0 else event1["data"]).replace("\"", ""))
        mentions0 = list(filter(lambda x: x.id == select_id0, b0))
    else:
        st.write("no agent 0 selection")
    if event1s:
        event1 = event1s[-1]
        select_id1 = int((event0["data"] if event1["agent"] == 0 else event1["data"]).replace("\"", ""))
        mentions1 = list(filter(lambda x: x.id == select_id1, b1))
    else:
        st.write("no agent 1 selection")

    st.write("SUCCESS" if reward else "FAILURE")

    visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)
    visualize_dialogue(dialogue)

id2dialogueidx = {
    x["chat_id"]: i
    for i, x in enumerate(finished_dialogues)
}
with open('../../aaai2020/experiments/data/onecommon/shared_4.json', "r") as f:
    scenario_list = json.load(f)
boards = {
    scenario['uuid']: scenario
    for scenario in scenario_list
}

idx = st.number_input("Train dialogue number", 0, len(finished_dialogues))
process_dialogue(finished_dialogues[idx])
