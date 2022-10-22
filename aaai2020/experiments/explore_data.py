from hfdata import corpus
from collections import defaultdict

def chats_per_scenario(dataset):
    scenario_to_chat = defaultdict(set)
    for example in dataset:
        scenario_to_chat[example.scenario_id].add(example.chat_id)
    return max(map(len, scenario_to_chat.values()))

datasets = [corpus.train, corpus.valid]
for dataset in datasets:
    print(chats_per_scenario(dataset))
