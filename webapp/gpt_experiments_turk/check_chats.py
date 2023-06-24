import json                                                  
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

with Path("all.json").open("r") as f:
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
    and None not in chat["workers"]
]


def compute_wins(chats):
    wins = Counter()
    total = Counter()
    turns = defaultdict(list)
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

    return wins, total, turns


wins, totals, turns = compute_wins(ok_selected_chats)

gpt_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
    and len(chat["dialogue"]) > 2
    and "gpt" in list(chat["agent_types"].values())
]
pragmatic_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
    and len(chat["dialogue"]) > 2
    and "pragmatic_confidence" in list(chat["agent_types"].values())
]
human_selected_chats = [
    chat for chat in chats
    if chat["num_players_selected"] == 2
    #and len(chat["dialogue"]) > 2
    and "gpt" not in list(chat["agent_types"].values())
    and "pragmatic_confidence" not in list(chat["agent_types"].values())
    and None not in chat["workers"]
]

print(any(["AKQAI78JTXXC9" in chat["workers"] for chat in chats]))

#print(compute_wins(human_selected_chats))
print("wins")
print(wins)
print("totals")
print(totals)

print("turns")
for k,v in turns.items():
    print(k, np.mean(v))



import pdb; pdb.set_trace()
