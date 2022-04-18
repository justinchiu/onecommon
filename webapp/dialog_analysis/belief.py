import streamlit as st
import streamlit.components.v1 as components

from functools import partial

import random
import json
from functools import partial
import numpy as np
from scipy.special import logsumexp as lse

#random.seed(1234)
#np.random.seed(1234)

class Dot:
    def __init__(self, item):
        for k,v in item.items():
            setattr(self, k, v)

    def html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size
        f = self.color
        label = f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id}</text>'
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" /> {label}'

    def select_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 2
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="red" stroke-width="3" stroke-dasharray="3,3"  />'

    def intersect_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 4
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="blue" stroke-width="3" stroke-dasharray="3,3"  />'

    def __repr__(self):
        return f"Dot {self.id}: ({self.x}, {self.y}) r={self.size} f={self.color}"


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

def visualize_board(left_dots, right_dots, select0, select1, intersect):
    shift = 430

    left_dots_html = map(lambda x: x.html(), left_dots)
    right_dots_html = map(lambda x: x.html(shift), right_dots)

    intersect_dots = map(lambda x: x.intersect_html(), intersect)

    select_left = list(filter(lambda x: int(x.id) == select0, left_dots))[0]
    select_right = list(filter(lambda x: int(x.id) == select1, right_dots))[0]

    print("left dots")
    print(left_dots)
    print("right dots")
    print(right_dots)
    print("intersect")
    print(intersect)

    left_ids = np.array([x.id for x in left_dots])
    intersect_ids = np.array([x.id for x in intersect])
    s = np.array([1 if x in intersect_ids else 0 for x in left_ids])

    print("left ids")
    print(left_ids)
    print("intersect ids")
    print(intersect_ids)
    print("state")
    print(s)

    # VERY PESSIMISTIC PRIOR
    # actually, this is incorrect.
    # if we know 6/7 overlap, the marginal dist should be 1/7 not included
    # given K overlap, marginal dist is 1 - 6Ck / 7Ck = k/7
    # guess we should not assume K overlap though, be even dumber?
    state_prior = np.ones((7,)) / 2
    print("prior")
    print(state_prior)
    state = np.stack((state_prior, 1-state_prior), 1)

    ask = np.array([1 if x in [2,5] else 0 for x in range(7)])
    print(ask)

    correct = 0.9
    error = 1 - correct
    likelihood = np.ones((2,2,2)) * error
    # ask about something, get correct answer
    likelihood[1,1,1] = correct
    likelihood[0,1,0] = correct
    # if you dont ask about something, no change
    likelihood[:,0] = 1
    print(likelihood)
    response = np.array([1 if x in [2,5] else 0 for x in range(7)])

    # RESPONSE IS FOR ALL DOTS INDEPENDENTLY
    def p_response(prior, ask):
        return (likelihood[:,ask] * prior).sum(-1).T

    def posterior(prior, ask, response):
        f = likelihood[response, ask] * prior
        return f / f.sum(-1, keepdims=True)

    def entropy(px):
        return -(px * np.log(px)).sum(-1)

    def info_gain(prior, ask, response):
        Hs = entropy(prior)
        Hs_r = entropy(posterior(prior, ask, response))
        return (Hs - Hs_r)[ask.astype(bool)].sum()

    def expected_info_gain(prior, ask, p_response):
        Hs = entropy(prior)
        r0 = np.zeros((7,), dtype=int)
        r1 = np.ones((7,), dtype=int)
        Hs_r0 = entropy(posterior(prior, ask, r0))
        Hs_r1 = entropy(posterior(prior, ask, r1))
        EHs_r = (p_response * np.stack((Hs_r0, Hs_r1), 1)).sum(-1)
        return (Hs - EHs_r)[ask.astype(bool)].sum()

    p_s_ar = posterior(state, ask, response)
    dH = info_gain(state, ask, response)

    p_r = p_response(state, ask)
    EdH = expected_info_gain(state, ask, p_r)

    asks = [np.unpackbits(np.array([x], dtype=np.ubyte))[1:] for x in range(1, 128)]
    EdHs = []
    for ask in asks:
        p_r = p_response(state, ask)
        EdH = expected_info_gain(state, ask, p_r)
        EdHs.append(EdH)
    EdHs = np.array(EdHs)

    # RESPONSES ARE IN PARALLEL, SO ASK ABOUT ALL 7 DOTS
    # RESPONSE IS FOR ALL DOTS JOINTLY
    def p_response(prior, ask):
        f = likelihood[:, ask] * prior
        f = f[:,ask.astype(bool),:].prod(1)
        return f.sum(-1).T

    def posteriors(prior, ask):
        f = likelihood[:, ask] * prior
        f = f[:,ask.astype(bool),:].prod(1)
        # returns normalized f[response, state] = p(state | ask, response)
        return f / f.sum(-1, keepdims=True)

    def info_gain(prior, ask, response):
        Hs = entropy(prior)
        Hs_r = entropy(posterior(prior, ask, response))
        return (Hs - Hs_r)[ask.astype(bool)].sum()

    def expected_info_gain(prior, ask, p_response):
        Hs = entropy(prior)
        r0 = np.zeros((7,), dtype=int)
        r1 = np.ones((7,), dtype=int)
        Hs_r0 = entropy(posterior(prior, ask, r0))
        Hs_r1 = entropy(posterior(prior, ask, r1))
        EHs_r = (p_response * np.stack((Hs_r0, Hs_r1), 1)).sum(-1)
        return (Hs - EHs_r)[ask.astype(bool)].sum()

    p_s_ar = posterior(state, ask, 1)
    dH = info_gain(state, ask, 1)

    p_r = p_response(state, ask)
    EdH = expected_info_gain(state, ask, p_r)

    asks = [np.unpackbits(np.array([x], dtype=np.ubyte))[1:] for x in range(1, 128)]
    EdHs = []
    for ask in asks:
        p_r = p_response(state, ask)
        EdH = expected_info_gain(state, ask, p_r)
        EdHs.append(EdH)
    EdHs = np.array(EdHs)
    import pdb; pdb.set_trace()

    # We want to ask about particular dot STRUCTURES

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

"""
train_json_file = "../../aaai2020/experiments/data/onecommon/final_transcripts.json"
with open(train_json_file, "r") as f:
    train_dialogues = json.load(f)
id2dialogue = {x["scenario_uuid"]: x for x in train_dialogues}
"""

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


#dialogues_type = st.selectbox("Human or robot dialogue", options=["robot", "human"])
#st.write(f"Dialogue idxs: {dialogue_idxs['human' if dialogues_type == 'human' else 'pragmatic_confidence']}")
#dialogues = human if dialogues_type == "human" else robot
#dialogue_id = st.select_slider("Dialogue number", options=list(range(len(dialogues))))

dialogues = robot
dialogue_id = 3

process_dialogue(dialogues[dialogue_id])


