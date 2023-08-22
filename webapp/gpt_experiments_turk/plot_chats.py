import numpy as np
import pandas
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mtick
import nltk
#to_print = list(reversed(['uabaseline_same_opt', 'pragmatic_confidence', 'human']))
to_print = list(reversed(['gpt', 'pragmatic_confidence', 'human']))
# to_print = list(reversed(['uabaseline_same_opt', 'human']))
# to_print = list(reversed(['human']))

def tokenize_and_filter(utterance):
    punct = set(".,\"'!-")
    toks = nltk.word_tokenize(utterance)
    return [tok for tok in toks if tok not in punct]

def human_dialogues(chat):
    dialogue = chat['dialogue']
    human_agent_numbers = set([int(k) for k, v in chat['agent_types'].items()
                           if v == 'human'])
    return [tokenize_and_filter(d) for ix, d in dialogue if ix in human_agent_numbers]

if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chat_json_file")
    parser.add_argument("--min_successful_games", type=int, default=0)
    parser.add_argument("--min_completed_games", type=int, default=1)
    parser.add_argument("--min_success_type", choices=['all_games', 'human_human_games'], default="human_human_games")
    args = parser.parse_args()
    with open(args.chat_json_file, 'r') as f:
        chats = json.load(f)

    # FILTER OUT HUMAN GAMES WITHOUT 2 TURKERS
    chats = [
        chat for chat in chats
        if ("gpt" in chat["agent_types"].values() and None not in chat["workers"])
        or ("pragmatic_confidence" in chat["agent_types"].values() and None not in chat["workers"])
        or (len(chat["workers"]) == 2 and None not in chat["workers"])
        #or len(chat["workers"]) == 2
    ]


    workers = set([worker for chat in chats for worker in chat["workers"]])

    # associate games with workers
    worker2games = defaultdict(list)
    for chat in chats:
        for worker in chat["workers"]:
            worker2games[worker].append(chat)

    worker2success = {worker: Counter() for worker in workers}
    worker2total = {worker: Counter() for worker in workers}
    for worker, games in worker2games.items():
        for game in games:
            partner = "human"
            agent_types = game["agent_types"].values()
            if "gpt" in agent_types:
                partner = "ours"
            if "pragmatic_confidence" in agent_types:
                partner = "baseline"

            if game["outcome"]["reward"] == 1:
                worker2success[worker][partner] += 1
            worker2total[worker][partner] += 1

    worker2rate = {
        worker: {
            key: worker2success[worker][key] / worker2total[worker][key]
            for key in worker2success[worker].keys()
        }
        for worker in workers
    }
    import pdb; pdb.set_trace()

