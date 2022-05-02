import streamlit as st
import streamlit.components.v1 as components

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
json_file = "../../aaai2020/experiments/data/onecommon/final_transcripts.json"

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
    st.table(dialogue)

def process_dialogue(dialogue_dict):
    id = dialogue_dict["uuid"]
    scenario_id = dialogue_dict["scenario_uuid"]
    dialogue = get_dialogue(dialogue_dict)
    board = dialogue_dict["scenario"]
    agent_types = dialogue_dict["agents"]

    st.write(f"Chat scenario id: {scenario_id}")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]

    reward = dialogue_dict["outcome"]["reward"]
    event0 = dialogue_dict["events"][-2]
    event1 = dialogue_dict["events"][-1]
    select_id0 = int((event0["data"] if event0["agent"] == 0 else event1["data"]).replace("\"", ""))
    select_id1 = int((event0["data"] if event1["agent"] == 0 else event1["data"]).replace("\"", ""))
    # set mentions = [selections]
    mentions0 = list(filter(lambda x: x.id == select_id0, b0))
    mentions1 = list(filter(lambda x: x.id == select_id1, b1))

    #turn = st.radio("Turn number", np.arange(len(dialogue)))
    #turn = st.number_input("Turn number", 0, len(dialogue)-1)

    #st.header("Dialogue so far")
    #for t in range(turn):
        #st.write(f"u: {dialogue[t]['utterance_language']} || r: {dialogue[t]['response_language']}")

    #turn = dialogue[turn]
    st.write("SUCCESS" if reward else "FAILURE")

    visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)
    visualize_dialogue(dialogue)

id2dialogueidx = {
    x["uuid"]: i
    for i, x in enumerate(finished_dialogues)
}

#st.write(f"Dialogue idxs: {dialogue_idxs}")
#dialogue_id = st.select_slider("Dialogue number", options=list(range(len(dialogue_idxs))))

# access any dialogue
#dialogue_id = st.text_input("Dialogue uuid", value="C_492a4d5a0195493b8f8ee4f0fbe5ab8d")
#process_dialogue(finished_dialogues[id2dialogueidx[dialogue_id]])

# inspect train / valid partner models

idx = st.select_slider("Train dialogue number", options=list(range(len(finished_dialogues))))
process_dialogue(finished_dialogues[idx])
