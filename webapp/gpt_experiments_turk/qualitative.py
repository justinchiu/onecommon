from pathlib import Path
import json

from oc.dot import Dot, visualize_board, single_board_html

scenarios_path = Path("../../aaai2020/experiments/data/onecommon/shared_4.json")
with scenarios_path.open("r") as f:
    scenarios = json.load(f)

dial_path = Path("blah.json")
with dial_path.open("r") as f:
    dials = json.load(f)

turk_gpt = [
    dial for dial in dials
    if "gpt" in dial["agent_types"].values()
    and len(dial["workers"]) > 0
]
turk_baseline = [
    dial for dial in dials
    if "pragmatic_confidence" in dial["agent_types"].values()
    and len(dial["workers"]) > 0
]


gpt_idx = 9

def print_chat(chat):
    color_map = {
        int(k): "blue" if v != "human" else "red"
        for k,v in chat["agent_types"].items()
    }
    for speaker, text in chat["dialogue"]:
        color = color_map[speaker]
        print(f"{{\color{{{color}}} {chat['agent_types'][str(speaker)]}}}:\;& {text}\\\\")

print(turk_gpt[gpt_idx]["scenario_id"])
print_chat(turk_gpt[gpt_idx])
print(turk_gpt[gpt_idx]["outcome"])

baseline_idx = 15
print(turk_baseline[baseline_idx]["scenario_id"])
print_chat(turk_baseline[baseline_idx])
print(turk_baseline[baseline_idx]["outcome"])
import pdb; pdb.set_trace()
