import json                                                  
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

with Path("blah.json").open("r") as f:
    chats = json.load(f)

zero_chats = [chat for chat in chats if chat["outcome"]["reward"] == 0]
# chats that had an error
neg_chats = [chat for chat in chats if chat["outcome"]["reward"] == -1]
win_chats = [chat for chat in chats if chat["outcome"]["reward"] == 1]


selected_chats = [chat for chat in chats if chat["num_players_selected"] == 2]


def dialogue_lens(chats):
    return np.array([len(x["dialogue"]) for x in chats])

print(dialogue_lens(zero_chats))
print(dialogue_lens(neg_chats))
print(dialogue_lens(win_chats))

sel_lens = dialogue_lens(selected_chats)
print(sel_lens)

ok_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
]


def compute_wins(chats):
    wins = Counter()
    total = Counter()
    turns = defaultdict(list)
    words = defaultdict(list)
    humanwords = defaultdict(list)
    for chat in chats:
        # get system
        agent_types = list(chat["agent_types"].values())

        system = "human"
        if "gpt" in agent_types:
            system = "gpt"
        if "pragmatic_confidence" in agent_types:
            system = "pragmatic_confidence"

        total[system] += 1
        if chat["outcome"]["reward"] == 1:
            wins[system] += 1

        turns[system].append(len(chat["dialogue"]))
        for speaker, turn in chat["dialogue"]:
            words[system].append(len(turn.strip().split()))
            if agent_types[speaker] == "human":
                humanwords[system].append(len(turn.strip().split()))
    return wins, total, turns, words, humanwords


wins, totals, turns, words, humanwords = compute_wins(ok_selected_chats)

gpt_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
    and "gpt" in list(chat["agent_types"].values())
]
pragmatic_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
    and "pragmatic_confidence" in list(chat["agent_types"].values())
]
human_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
    and "gpt" not in list(chat["agent_types"].values())
    and "pragmatic_confidence" not in list(chat["agent_types"].values())
]

total_non_expert = len([c["outcome"]["reward"] for c in human_selected_chats if None not in c["workers"]])
win_non_expert = sum([c["outcome"]["reward"] for c in human_selected_chats if None not in c["workers"]])
win_expert = sum([c["outcome"]["reward"] for c in human_selected_chats if None in c["workers"]])
total_expert = len([c["outcome"]["reward"] for c in human_selected_chats if None in c["workers"]])

print("Win rate for non expert", win_non_expert / total_non_expert)
print("Win rate for expert", win_expert / total_expert)

print("num unique workers", len(set([worker for chat in chats for worker in chat["workers"]])))

print("avg num turns")
for k,v in turns.items():
    print(k, np.mean(v))

print("avg num words per turn")
for k,v in words.items():
    print(k, np.mean(v))

print("avg num human words per turn")
for k,v in humanwords.items():
    print(k, np.mean(v))

print("median human words per turn")
for k,v in humanwords.items():
    print(k, np.median(v))


print("non-expert stuff")
non_expert_chats = [c for c in human_selected_chats if None not in c["workers"]]
nwins, ntotals, nturns, nwords, nhumanwords = compute_wins(non_expert_chats)
print("avg num turns")
for k,v in nturns.items():
    print(k, np.mean(v))

print("avg num words per turn")
for k,v in nwords.items():
    print(k, np.mean(v))

print("avg num human words per turn")
for k,v in nhumanwords.items():
    print(k, np.mean(v))

print("expert stuff")
expert_chats = [c for c in human_selected_chats if None in c["workers"]]
nwins, ntotals, nturns, nwords, nhumanwords = compute_wins(expert_chats)
print("avg num turns")
for k,v in nturns.items():
    print(k, np.mean(v))

print("avg num words per turn")
for k,v in nwords.items():
    print(k, np.mean(v))

print("avg num human words per turn")
for k,v in nhumanwords.items():
    print(k, np.mean(v))

import pdb; pdb.set_trace()

