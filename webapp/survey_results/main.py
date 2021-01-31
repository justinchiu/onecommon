import json
from functools import partial

import numpy as np

json_file = "experiments_nov-22/all_dialogues.json"

with open(json_file, "r") as f:
    dialogues = json.load(f)

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

#def get_survey_results(xs):
survey_qs = set(dialogues[0]["survey_result"].keys()) - {"has_survey", "comments"}

# completed games with surveys
dialogues_by_ty = chop_up([
    x for x in dialogues
    if x["num_players_selected"] == 2 and
        x["survey_result"]["has_survey"]
])

def survey_results(xs):
    d = {}
    for q in survey_qs:
        d[q] = np.array([x["survey_result"][q] for x in xs])
    return d

dsurvey = apply(dialogues_by_ty, survey_results)

def qmean(dxs):
    return {
        model_ty: {q: results.mean() for q, results in survey_results.items()}
        for model_ty, survey_results in dxs.items()
    }

def qvar(dxs):
    return {
        model_ty: {q: results.var() for q, results in survey_results.items()}
        for model_ty, survey_results in dxs.items()
    }

survey_means = qmean(dsurvey)
survey_vars = qvar(dsurvey)

"""
for model_ty, res in survey_means.items():
    print(model_ty)
    for q, mu in res.items():
        print(q, mu)
    print()

"""

print(len(get_complete(dialogues)))
