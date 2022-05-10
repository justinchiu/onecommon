from pathlib import Path

#import streamlit as st
#import streamlit.components.v1 as components

from functools import partial

import random
import json
from functools import partial
import numpy as np
from scipy.special import logsumexp as lse

from dot import Dot

#random.seed(1234)
#np.random.seed(1234)

def classify_sets(xs, ys):
    # with MAP ref resolution, ambiguous = incorrect
    ambiguous = xs != ys
    unresolvable = len(ys) == 0
    specific = xs == ys
    return ambiguous, unresolvable, specific

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

analysis_path = Path("../../aaai2020/experiments/analysis_log")
scenarios = [f.stem for f in analysis_path.iterdir() if f.is_file()]

prior_ambiguous, prior_unresolvable, prior_specific = 0,0,0
plan_ambiguous, plan_unresolvable, plan_specific = 0,0,0
prior_num_turns = 0
plan_num_turns = 0
agreed = 0

for scenario in scenarios:
    with (analysis_path / scenario).with_suffix(".json").open() as f:
        turns = json.load(f)
        dialogue = dialogues[scenario]
        board = boards[scenario]

        dots = board["kbs"]
        dot_ids = np.array([[int(d["id"]) for d in ds] for ds in dots])

        for turn in turns:
            agent_id = turn["agent_id"]

            our_dots = dot_ids[0 if agent_id == 0 else 1]
            their_dots = dot_ids[0 if agent_id != 0 else 1]

            prior_plan = (np.array(turn["prior_plan"]).any(0)[0]
                if turn["prior_plan"] is not None else None)
            plan3_plan = (np.array(turn["plan3_plan"]).any(0)[0]
                if turn["plan3_plan"] is not None else None)
            prior_partner_ref = (np.array(turn["prior_partner_ref"]).any(0)[0]
                if turn["prior_partner_ref"] is not None else None)
            plan3_partner_ref = (np.array(turn["plan3_partner_ref"]).any(0)[0]
                if turn["plan3_partner_ref"] is not None else None)

            prior_dots = set(our_dots[prior_plan]) if prior_plan is not None else None
            plan_dots = set(our_dots[plan3_plan]) if plan3_plan is not None else None
            prior_partner_dots = set(their_dots[prior_partner_ref]) if prior_partner_ref is not None else None
            plan_partner_dots = set(their_dots[plan3_partner_ref]) if plan3_partner_ref is not None else None

            # typo in key, trailing space
            prior_lang = turn["prior_mentions_language "]
            plan_lang = turn["plan3_mentions_language "]

            if prior_dots is not None:
                a,u,s = classify_sets(prior_dots, prior_partner_dots)
                prior_ambiguous += a
                prior_unresolvable += u
                prior_specific += s
                prior_num_turns += 1

            if plan_dots is not None:
                a,u,s = classify_sets(plan_dots, plan_partner_dots)
                plan_ambiguous += a
                plan_unresolvable += u
                plan_specific += s
                plan_num_turns += 1

print("Prior")
print(f"A: {prior_ambiguous}, U: {prior_unresolvable}, S: {prior_specific}")
print(f"Prior num turns: {prior_num_turns}")
print("Plan")
print(f"A: {plan_ambiguous}, U: {plan_unresolvable}, S: {plan_specific}")
print(f"Plan num turns: {plan_num_turns}")
print(f"Num agreed: {agreed}")
