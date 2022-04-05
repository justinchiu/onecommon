import streamlit as st
import streamlit.components.v1 as components

from functools import partial

import random
import json
from functools import partial
import numpy as np

random.seed(1234)
np.random.seed(1234)

# Testing streamlit
st.title("OneCommon Training Data Visualizations")

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

finished_dialogues = get_selected(dialogues)

num_turns_mean = np.mean([len(get_dialogue(x)) for x in finished_dialogues])
num_turns_med = np.median([len(get_dialogue(x)) for x in finished_dialogues])
num_turns_std = np.std([len(get_dialogue(x)) for x in finished_dialogues])

utt_len_mean = np.mean([
    len(y.split())
    for x in finished_dialogues
    for y in get_dialogue(x)
])
utt_len_std = np.std([
    len(y.split())
    for x in finished_dialogues
    for y in get_dialogue(x)
])
utt_len_max = np.max([
    len(y.split())
    for x in finished_dialogues
    for y in get_dialogue(x)
])
print("Num turns mean")
print(num_turns_mean)
print("Num turns std")
print(num_turns_std)
print("Utt len mean")
print(utt_len_mean)
print("Utt len std")
print(utt_len_std)

# visualize dots
def dot_html(item, shift=0):
    x = item["x"] + shift
    y = item["y"]
    r = item["size"]
    f = item["color"]
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" />'

def select_html(item, shift=0):
    x = item["x"] + shift
    y = item["y"]
    r = item["size"] + 2
    f = item["color"] # ignored
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="red" stroke-width="2" stroke-dasharray="3,3"  />'

def visualize_board(board, select0, select1):
    shift = 430
    left_dots = board["kbs"][0]
    right_dots = board["kbs"][1]

    left_dots_html = map(dot_html, left_dots)
    right_dots_html = map(partial(dot_html, shift=shift), right_dots)
    select_left = list(filter(lambda x: int(x["id"]) == select0, left_dots))[0]
    select_right = list(filter(lambda x: int(x["id"]) == select1, right_dots))[0]
    nl = "\n"
    html = f"""
    <svg width="860" height="430">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(left_dots_html)}
    {select_html(select_left)}
    <circle cx="645" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(right_dots_html)}
    {select_html(select_right, shift=shift)}
    </svg>
    """
    components.html(html, height=430, width=860)

def visualize_dialogue(dialogue):
    st.table(dialogue)

dialogue_idxs = np.random.choice(len(finished_dialogues), 20, replace=False)

subsampled = [finished_dialogues[idx] for idx in dialogue_idxs]

# we only care about 'human' and 'pragmatic_confidence'
def process_dialogue(dialogue_dict):
    import pdb; pdb.set_trace()
    scenario_id = dialogue_dict["scenario_id"]
    dialogue = dialogue_dict["dialogue"]
    agent_types = dialogue_dict["agent_types"]

    reward = dialogue_dict["outcome"]["reward"]
    event0 = dialogue_dict["events"][-2]
    event1 = dialogue_dict["events"][-1]
    select0 = int((event0["data"] if event0["agent"] == 0 else event1["data"]).replace("\"", ""))
    select1 = int((event0["data"] if event1["agent"] == 0 else event1["data"]).replace("\"", ""))

    #board = boards[scenario_id]
    import pdb; pdb.set_trace()
    board = dialogue_dict[""]
    st.write("SUCCESS" if reward else "FAILURE")
    visualize_board(board, select0, select1)
    st.write(f"Agent 0: {agent_types['0']} || 1: {agent_types['1']}")
    visualize_dialogue(dialogue)


st.write(f"Dialogue idxs: {dialogue_idxs}")
dialogue_id = st.select_slider("Dialogue number", options=list(range(len(dialogue_idxs))))

process_dialogue(subsampled[dialogue_id])
