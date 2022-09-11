from pathlib import Path

#import streamlit as st
#import streamlit.components.v1 as components

from functools import partial

from collections import Counter, defaultdict

import random
import json
from functools import partial
import numpy as np
from scipy.special import logsumexp as lse

from dot import Dot

#random.seed(1234)
#np.random.seed(1234)

# OPTIONS
FIRST_TURN_ONLY = False
ABSOLUTE = True

def classify_sets(xs, ys):
    # with MAP ref resolution, ambiguous = incorrect
    ambiguous = xs != ys
    unresolvable = ys is None or len(ys) == 0
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

split = "train"
split = "valid_1"
if ABSOLUTE:
    split = "valid_1_absolute"
#split = "valid_1_absolute_or_collapsed"
split = "valid_1_absolute_cost_collapsed"
#split = "valid_1_absolute_cost"
#split = "valid_1_absolute_cost_egocentric_collapsed"
analysis_path = Path("../../aaai2020/experiments/analysis_log") / split
scenarios = [f.stem for f in analysis_path.iterdir() if f.is_file()]

prior_ambiguous, prior_unresolvable, prior_specific = 0,0,0
plan_ambiguous, plan_unresolvable, plan_specific = 0,0,0
prior_num_turns = 0
plan_num_turns = 0
prior_zeros, plan_zeros = 0,0

labels_prior = {x: Counter() for x in range(4,7)}
#labels_ablate = {x: Counter() for x in range(4,7)}
labels = {x: Counter() for x in range(4,7)}

mistakes = defaultdict(list)

num_scenarios = Counter()
num_turns = 0

for scenario in scenarios:
    with (analysis_path / scenario).with_suffix(".json").open() as f:
        turns = json.load(f)
        # FIRST TURN ONLY
        if FIRST_TURN_ONLY:
            turns = [turns[0]]
        dialogue = dialogues[scenario]
        board = boards[scenario]

        dots = board["kbs"]
        dot_ids = np.array([[int(d["id"]) for d in ds] for ds in dots])
        intersection_size = len(set(dot_ids[0]).intersection(set(dot_ids[1])))
        num_scenarios[intersection_size] += 1


        for turn_i, turn in enumerate(turns):
            num_turns += 1
            agent_id = turn["writer_id"]

            # PLAN FEATURE LABELS
            if "label_prior" in turn:
                labels_prior[intersection_size][turn["label_prior"]] += 1
                labels[intersection_size][turn["label"]] += 1
                if turn["label"] == 0:
                    mistakes[scenario].append(turn_i)
            # / PLAN FEATURE LABELS

            our_dots = dot_ids[0 if agent_id == 0 else 1]
            their_dots = dot_ids[0 if agent_id != 0 else 1]

            prior_plan = (np.array(turn["prior_plan"]).any(0)[0]
                if (turn["prior_plan"] is not None and len(turn["prior_plan"][0]) > 0) else None)
            plan_plan = (np.array(turn["plan_plan"]).any(0)[0]
                if turn["plan_plan"] is not None else None)
            prior_partner_ref = (np.array(turn["prior_partner_ref"]).any(0)[0]
                if turn["prior_partner_ref"] is not None else None)
            plan_partner_ref = (np.array(turn["plan_partner_ref"]).any(0)[0]
                if turn["plan_partner_ref"] is not None else None)

            prior_dots = set(our_dots[prior_plan]) if prior_plan is not None else None
            plan_dots = set(our_dots[plan_plan]) if plan_plan is not None else None
            prior_partner_dots = set(their_dots[prior_partner_ref]) if prior_partner_ref is not None else None
            plan_partner_dots = set(their_dots[plan_partner_ref]) if plan_partner_ref is not None else None

            prior_lang = turn["prior_mentions_language"]
            plan_lang = turn["plan_mentions_language"]

            if prior_dots is not None and prior_plan.sum() > 0:
                a,u,s = classify_sets(prior_dots, prior_partner_dots)
                prior_ambiguous += a
                prior_unresolvable += u
                prior_specific += s
                prior_num_turns += 1
            else:
                prior_zeros += 1

            if plan_dots is not None and plan_plan.sum() > 0:
                a,u,s = classify_sets(plan_dots, plan_partner_dots)
                plan_ambiguous += a
                plan_unresolvable += u
                plan_specific += s
                plan_num_turns += 1
            else:
                prior_zeros += 1

print("Prior")
print(f"A: {prior_ambiguous}, U: {prior_unresolvable}, S: {prior_specific}")
print(f"Prior num turns: {prior_num_turns}, num zeros: {prior_zeros}")
print("Plan")
print(f"A: {plan_ambiguous}, U: {plan_unresolvable}, S: {plan_specific}")
print(f"Plan num turns: {plan_num_turns}, num zeros: {plan_zeros}")


def print_labels(labels):
    for k in labels.keys():
        print(f"Intersect size: {k}")
        print(f"E: {labels[k][0]}, C: {labels[k][1]}, U: {labels[k][2]}, S: {labels[k][3]}")
    sum_labels = labels[4] + labels[5] + labels[6]
    print(f"Total")
    print(f"E: {sum_labels[0]}, C: {sum_labels[1]}, U: {sum_labels[2]}, S: {sum_labels[3]}")

print()
print("PLAN FEATURE EVALUATION")
print("Prior")
print_labels(labels_prior)
#print("Ablate")
#print_labels(labels_ablate)
print("Label")
print_labels(labels)

print("Num scenarios by intersection size")
print(num_scenarios)

print(f"Num turns: {num_turns}")

print(mistakes)
import pdb; pdb.set_trace()
