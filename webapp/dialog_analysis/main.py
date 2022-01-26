import streamlit as st
import streamlit.components.v1 as components

from functools import partial

import json
from functools import partial
import numpy as np

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

def visualize_board(board):
    left_dots = board["kbs"][0]
    right_dots = board["kbs"][1]

    left_dots_html = map(dot_html, left_dots)
    right_dots_html = map(partial(dot_html, shift=430), right_dots)
    nl = "\n"
    html = f"""
    <svg width="860" height="430">
    <circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(left_dots_html)}
    <circle cx="645" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>
    {nl.join(right_dots_html)}
    </svg>
    """
    components.html(html, height=430, width=860)

def visualize_dialogue(dialogue):
    st.table(dialogue)

subsampled_by_ty = {
    k: np.random.choice(v, 10, replace=False)
    for k,v in dialogues_by_ty.items()
}

# we only care about 'human' and 'pragmatic_confidence'
human = subsampled_by_ty["human"]
robot = subsampled_by_ty["pragmatic_confidence"]

def process_dialogue(dialogue_dict):
    scenario_id = dialogue_dict["scenario_id"]
    dialogue = dialogue_dict["dialogue"]
    agent_types = dialogue_dict["agent_types"]

    board = boards[scenario_id]
    visualize_board(board)
    st.write(f"Agent 0: {agent_types['0']} || 1: {agent_types['1']}")
    visualize_dialogue(dialogue)


dialogues_type = st.select_slider("Human or robot dialogue", options=["human", "robot"])
dialogues = human if dialogues_type == "human" else robot
dialogue_id = st.select_slider("Dialogue number", options=list(range(len(dialogues))))

process_dialogue(dialogues[dialogue_id])
