from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from functools import partial

import random
import json
from functools import partial
import numpy as np

from dot import Dot, visualize_board
from response import ResponseDB

random.seed(1234)
np.random.seed(1234)

# Testing streamlit
st.title("OneCommon Partner Response Visualizations")

#json_file = "experiments_nov-22/all_dialogues.json"
json_file = "../aaai2020/experiments/data/onecommon/final_transcripts.json"

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

def get_intersect(board):
    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]
    return intersect0, intersect1

def intersect_size(board):
    i0, i1 = get_intersect(board)
    assert len(i0) == len(i1)
    return len(i0)

def process_dialogue(dialogue_dict, mentions, db):
    # mentions: [(Agent id, [[Dot id]])]
    id = dialogue_dict["uuid"]
    scenario_id = dialogue_dict["scenario_uuid"]
    dialogue = get_dialogue(dialogue_dict)
    board = dialogue_dict["scenario"]
    agent_types = dialogue_dict["agents"]

    st.write(f"Chat id: {id}")
    st.write(f"Chat scenario id: {scenario_id}")

    b0 = [Dot(x) for x in board["kbs"][0]]
    b1 = [Dot(x) for x in board["kbs"][1]]
    intersect0 = [x for x in b0 for y in b1 if x.id == y.id]
    intersect1 = [x for x in b1 for y in b0 if x.id == y.id]

    reward = dialogue_dict["outcome"]["reward"]
    #event0 = dialogue_dict["events"][-2]
    #event1 = dialogue_dict["events"][-1]
    #select_id0 = int((event0["data"] if event0["agent"] == 0 else event1["data"]).replace("\"", ""))
    #select_id1 = int((event0["data"] if event1["agent"] == 0 else event1["data"]).replace("\"", ""))
    events = dialogue_dict["events"]
    event0 = [x["data"] for x in events if x["action"] == "select" and x["agent"] == 0]
    event1 = [x["data"] for x in events if x["action"] == "select" and x["agent"] == 1]
    # take last selection, if for some reason there are multiple
    # i think humans might be allowed to revise?
    select_id0 = [int(x.replace("\"", "")) for x in event0][-1]
    select_id1 = [int(x.replace("\"", "")) for x in event1][-1]
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

    #visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)
    visualize_dialogue(dialogue)

    num_turns = len(events)
    turn = st.number_input("Turn no", 0, num_turns-1)

    mentions0, mentions1 = None, None
    agent = events[turn]["agent"]
    if mentions is not None:
        assert agent == mentions[turn][0]
        turn_mention_ids = set([x for xs in mentions[turn][1] for x in xs])
        turn_mentions = [x for x in (b0 if agent == 0 else b1) if x.id in turn_mention_ids]

        mentions0 = turn_mentions if agent == 0 else None
        mentions1 = turn_mentions if agent == 1 else None

    visualize_board(b0, b1, mentions0, mentions1, intersect0, intersect1)
    LEFT = "left"
    RIGHT = "right"
    st.write(f"Prev utterance {LEFT if agent == 0 else RIGHT}")
    st.write(dialogue[turn])
    st.write(f"Response {RIGHT if agent == 0 else LEFT}")
    st.write(dialogue[turn+1])

    prev_results = db.get_id_turn(id, turn+1)
    if len(prev_results) > 0:
        st.write("Previous results")
        st.write(prev_results)
    else:
        st.write("No previous results")

    response = st.radio("Response is", [0,1,2], format_func=lambda x: db.R2S[x])
    if st.button("Submit"):
        # log to db
        # turn is the response turn
        # st.write(id, turn+1, 1-agent, response)
        db.add(id, turn+1, 1-agent, response)

def get_referent_ids(referentss, markabless, dialogue_id):
    if dialogue_id not in referentss or dialogue_id not in markabless:
        return None
    referents = referentss[dialogue_id]
    markables = markabless[dialogue_id]
    # referents: {M#: Markable}
    # markables: {markables: List[DetailedMarkable], text: Str}
    # return List[(agent, dot ids)]
    #chunked_text = markables["text"].split("\n")
    #start_idx = [0] + [len(x) for x in chunked_text]
    text = markables["text"]
    split_text = text.split("\n")
    markables = markables["markables"]
    idxs = [0]
    for i, char in enumerate(text):
        if char == "\n":
            idxs.append(i)
    idxs.append(len(text))

    num_turns = len(idxs) - 2

    markable_idx = 0

    mentions = []
    for turn_idx in range(len(split_text)):
        turn_start = idxs[turn_idx]
        turn_end = idxs[turn_idx+1]

        agent_text = int(split_text[turn_idx][0])
        mentions.append((agent_text, []))

        if markable_idx >= len(markables):
            # break out of outer loop
            break

        markable_start = markables[markable_idx]["start"]
        markable_end = markables[markable_idx]["end"]
        #while markable_start < turn_end and turn_start < markable_end:
        while turn_start <= markable_start and markable_end <= turn_end:
            #print(markable_start, markable_end)
            #print(turn_start, turn_end)

            markable_id = markables[markable_idx]["markable_id"]
            refs = referents[markable_id]["referents"]
            ref_ids = [int(x.split("_")[-1]) for x in refs]

            agent_markable = int(refs[0].split("_")[1])
            assert agent_text == agent_markable

            mentions[-1][1].append(ref_ids)

            markable_idx += 1
            if markable_idx >= len(markables):
                break
            markable_start = markables[markable_idx]["start"]
            markable_end = markables[markable_idx]["end"]

    return mentions

db = ResponseDB()

st.write("""Annotation instructions:\n
* Response = none if response does not directly confirm all dots
    asked about in the previous turn.
    Follow up questions should be marked as none.\n
* Response = confirm if all dots that were asked about in prev are
    confirmed\n
* Response = disconfirm if all dots that were asked about in prev are
    disconfirmed\n
""")

referent_path = Path("../aaai2020/annotation/aggregated_referent_annotation.json")
markable_path = Path("../aaai2020/annotation/markable_annotation.json")

with referent_path.open() as f:
    referents = json.load(f)
with markable_path.open() as f:
    markables = json.load(f)

id2dialogueidx = {
    x["uuid"]: i
    for i, x in enumerate(finished_dialogues)
}
intersect_size2d = {
    k: [x for x in finished_dialogues if intersect_size(x["scenario"]) == k]
    for k in range(4, 7)
}
intersect_size = st.number_input("Intersect size", 4, 6)
#intersect_size=4
sized_dialogues = intersect_size2d[intersect_size]

idx = st.select_slider("Train dialogue number", options=list(range(len(sized_dialogues))))
#idx = 1
dialogue = sized_dialogues[idx]
id = dialogue["uuid"]
mentions = get_referent_ids(referents, markables, id)
process_dialogue(dialogue, mentions, db)
