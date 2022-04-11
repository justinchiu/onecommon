import streamlit as st
import streamlit.components.v1 as components

from functools import partial

import random
import json
from functools import partial
import numpy as np

#random.seed(1234)
#np.random.seed(1234)

# Testing streamlit
st.title("OneCommon Visualizations")

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


num_turns_mean = apply(dialogues_by_ty, lambda xs: np.mean([len(x["dialogue"]) for x in xs]))
num_turns_med = apply(dialogues_by_ty, lambda xs: np.median([len(x["dialogue"]) for x in xs]))
num_turns_std = apply(dialogues_by_ty, lambda xs: np.std([len(x["dialogue"]) for x in xs]))

utt_len_mean = {}
utt_len_std = {}
utt_len_max = {}
for ty, xs in dialogues_by_ty.items():
    utt_len_mean[ty] = np.mean([
        len(utt.split())
        for x in xs for id, utt in x["dialogue"]
        if x["agent_types"][f"{id}"] == ty
    ])
    utt_len_std[ty] = np.std([
        len(utt.split())
        for x in xs for id, utt in x["dialogue"]
        if x["agent_types"][f"{id}"] == ty
    ])
    utt_len_max[ty] = np.max([
        len(utt.split())
        for x in xs for id, utt in x["dialogue"]
        if x["agent_types"][f"{id}"] == ty
    ])

"""
print("Num turns mean")
print(num_turns_mean)
print("Num turns std")
print(num_turns_std)
print("Utt len mean")
print(utt_len_mean)
print("Utt len std")
print(utt_len_std)
"""

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

dialogue_idxs = {
    k: np.random.choice(len(v), 20, replace=False)
    for k,v in finished_dialogues_by_ty.items()
}
dialogue_idxs = {
    "human": [96, 74, 42, 83, 138, 2, 160, 192, 202, 176, 196, 182, 180, 193, 148, 146, 58, 99, 41, 120, 92, 117],
    "pragmatic_confidence": [237, 29, 170, 131, 113, 92, 40, 65, 64, 74, 186, 71, 137, 190, 32, 66, 22, 111, 141, 59],
    "uabaseline_same_opt": dialogue_idxs["uabaseline_same_opt"],
}

subsampled_by_ty = {
    k: [v[idx] for idx in dialogue_idxs[k]]
    for k,v in finished_dialogues_by_ty.items()
}

# we only care about 'human' and 'pragmatic_confidence'
human = subsampled_by_ty["human"]
robot = subsampled_by_ty["pragmatic_confidence"]

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
    st.write("SUCCESS" if reward else "FAILURE")
    visualize_board(board, select0, select1)
    st.write(f"Agent 0: {agent_types['0']} || 1: {agent_types['1']}")
    visualize_dialogue(dialogue)


dialogues_type = st.selectbox("Human or robot dialogue", options=["robot", "human"])
st.write(f"Dialogue idxs: {dialogue_idxs['human' if dialogues_type == 'human' else 'pragmatic_confidence']}")
dialogues = human if dialogues_type == "human" else robot
dialogue_id = st.select_slider("Dialogue number", options=list(range(len(dialogues))))

process_dialogue(dialogues[dialogue_id])
