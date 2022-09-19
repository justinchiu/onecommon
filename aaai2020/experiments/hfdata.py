from typing import NamedTuple

from pathlib import Path
import numpy as np
from jinja2 import Template

from datasets import Dataset, load_dataset

from domain import get_domain

from corpora.data import Dictionary, get_tag
from corpora.reference_sentence import ReferenceSentenceCorpus

domain = get_domain("one_common")
data = 'data/onecommon'
fold_num = 1
freq_cutoff = 20
use_properties = True
use_pairwise_features = True
use_unordered_pairwise = True
use_extrema_features = False


corpus = ReferenceSentenceCorpus(
    domain, data,
    train='train_reference_{}.txt'.format(fold_num),
    valid='valid_reference_{}.txt'.format(fold_num),
    test='test_reference_{}.txt'.format(fold_num),
    freq_cutoff=freq_cutoff, verbose=True,
    max_instances_per_split=None,
    max_mentions_per_utterance=None,
    crosstalk_split=None,
    spatial_data_augmentation_on_train=False,
)

datadir = Path(data)

train_path = datadir / 'train_reference_{}.txt'.format(fold_num)
valid_path = datadir / 'valid_reference_{}.txt'.format(fold_num)
test_path = datadir / 'test_reference_{}.txt'.format(fold_num)

paths = {
    "train": train_path,
    "valid": valid_path,
    "test": test_path,
}
splits = paths.keys()

word_dict = Dictionary.from_file(str(train_path), freq_cutoff = freq_cutoff)

"""
REFERENT LAYOUT: [
    beginning idx,
    ending idx,
    utterance end idx,
    *binary indicators for occurrence, eg 7 dots 0 1 1 0 0 0 0 => dots 1 and 2
]
"""
def _split_referents(words, referent_idxs):
    stops = ["YOU:", "THEM:"]
    sents, current = [], []
    all_refs, current_refs = [], []
    split_ref_indices = []
    split_ref_objs = []
    assert len(referent_idxs) % 10 == 0
    for k in range(0, len(referent_idxs), 10):
        split_ref_indices.append(referent_idxs[k:k + 3])
        split_ref_objs.append(referent_idxs[k + 3:k + 10])
    split_ref_indices = np.array(split_ref_indices)
    ref_ix = 0
    for w in words:
        if w in stops:
            while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
                current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
                ref_ix += 1
            split_ref_indices[ref_ix:] -= len(current)
            if len(current) > 0:
                sents.append(current)
                all_refs.append(current_refs)
            current = []
            current_refs = []
        current.append(w)
    if len(current) > 0:
        while ref_ix < len(split_ref_indices) and split_ref_indices[ref_ix][-1] < len(current):
            current_refs.extend(list(split_ref_indices[ref_ix]) + split_ref_objs[ref_ix])
            ref_ix += 1
        sents.append(current)
        all_refs.append(current_refs)
    assert ref_ix == len(split_ref_indices)
    assert sum(len(refs) for refs in all_refs) == len(referent_idxs)
    assert len(all_refs) == len(sents)
    return sents, all_refs

class Conversation(NamedTuple):
    dots: list[float]
    sents: list[list[str]]
    refs: list[list[int]]
    partner_refs: list[list[int]]
    partner_refs_our_view: list[list[int]]

    scenario_id: str
    chat_id: str

    real_ids: list[str]
    partner_real_ids: list[str]


def get_conversations(split):
    conversations = []
    path = paths[split]
    with path.open() as f:
        for line in f:
            line = line.strip()

            tokens = line.split()

            # The linearized context values
            # Each dot is represented as (x, y, size, color)
            # There should be 28 values = 4 values * 7 dots.
            input_vals = [float(val) for val in get_tag(tokens, 'input')]

            words = get_tag(tokens, 'dialogue')
            word_idxs = word_dict.w2i(words)

            referent_idxs = [int(val) for val in get_tag(tokens, 'referents')]
            partner_referent_idxs = [int(val) for val in get_tag(tokens, 'partner_referents')]
            partner_referent_our_view_idxs = [int(val) for val in get_tag(tokens, 'partner_referents_our_view')]

            # i believe this is the final selection? need to verify
            output_idx = int(get_tag(tokens, 'output')[0])

            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            partner_real_ids = get_tag(tokens, 'partner_real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]

            ref_disagreement = list(map(int, get_tag(tokens, 'referent_disagreements')))
            partner_ref_disagreement = list(map(int, get_tag(tokens, 'partner_referent_disagreements')))

            sents, all_refs = _split_referents(words, referent_idxs)
            sents_, all_partner_refs = _split_referents(words, partner_referent_idxs)
            assert sents == sents_
            sents_, all_partner_refs_our_view = _split_referents(words, partner_referent_our_view_idxs)
            assert sents == sents_

            conversations.append(Conversation(
                sents = sents,
                refs = all_refs,
                partner_refs = all_partner_refs,
                partner_refs_our_view = all_partner_refs_our_view,

                dots = input_vals,

                scenario_id = scenario_id,
                chat_id = chat_id,

                real_ids = real_ids,
                partner_real_ids = partner_real_ids,
            ))
        return conversations

"""
Process conversations into examples.
Each conversation will be turned into num_turns examples (or num_turns / 2).
Each example will look like:
    * All previous turns: each turn is a list of tokens (strings)
    * All previous refs, partner refs, partner_refs_our_view
    * scenario, chat, and ids
Eg conversation prefixes
"""

def describe_dots(
    dots,
    use_pairwise_features = True,
    use_extrema_features = True,
):
    dots = np.array(dots).reshape(-1, 4)
    rounded_dots = (dots.round(2) * 100).astype(int)
    num_dots = dots.shape[0]
    # (x, y, size, color)
    dot_desc_template = Template(
        #"dot{{id}}: [x: {{x}}, y: {{y}}, size: {{size}}, color: {{color}}]"
        "dot{{id}} x: {{x}}, y: {{y}}, size: {{size}}, color: {{color}}"
    )
    description = " [SEP] ".join([
        dot_desc_template.render(
            id = i+1,
            x = rounded_dots[i, 0],
            y = rounded_dots[i, 1],
            size = rounded_dots[i, 2],
            color = rounded_dots[i, 3],
        ) for i in range(num_dots)
    ])

    dot_strings = [f"dot{i}" for i in range(1, 8)]
    if use_pairwise_features:
        # construct pairwise descriptions for each dot and 3 closest
        xy = dots[:,:2]
        dists = ((xy[:,None] - xy) ** 2).sum(-1)
        dists[range(7), range(7)] = dists.max() + 1
        closest_idxs = dists.argsort()[:,:2]

        #print(dists)
        #print(closest_idxs)
        #import pdb; pdb.set_trace()

        # ordered pairs
        dot_pairs = [(i, j) for i in range(7) for j in closest_idxs[i]]
        if use_unordered_pairwise:
            # unordered pairs
            dot_pairs = set([tuple(sorted(x)) for x in dot_pairs])

        pairwise_strs = []
        for i,j in dot_pairs:
            dot1 = dot_strings[i]
            dot2 = dot_strings[j]
            x1, y1, s1, c1 = dots[i]
            x2, y2, s2, c2 = dots[j]

            vert_comp = "above" if y1 > y2 else "below"
            hor_comp = "right" if x1 > x2 else "left"
            size_comp = "bigger" if s1 > s2 else "smaller"
            col_comp = "darker" if c1 > c2 else "lighter"

            vert_str = f"{dot1} is {vert_comp} {dot2}"
            hor_str = f"{dot1} is {hor_comp} of {dot2}"
            size_str = f"{dot1} is {size_comp} than {dot2}"
            col_str = f"{dot1} is {col_comp} than {dot2}"

            pairwise_strs.append(vert_str)
            pairwise_strs.append(hor_str)
            pairwise_strs.append(size_str)
            pairwise_strs.append(col_str)

        pairwise_str = ", ".join(pairwise_strs)
        description = f"{description} [SEP] {pairwise_str}"

    if use_extrema_features:
        pass
    return description

def describe_plan_dense(plan):
    num_dots = len(plan)
    dot_desc_template = Template("dot{{id}}: {{include}}")
    description = ", ".join([
        dot_desc_template.render(
            id = i+1,
            include = plan[i],
        ) for i in range(num_dots)
    ])
    return description

def describe_plan_sparse(plan):
    num_dots = len(plan)
    dot_desc_template = Template("dot{{id}}")
    description = ", ".join([
        f"dot{i+1}"
        for i in range(num_dots)
        if plan[i] == 1
    ])
    return description

# unused for now
class Example(NamedTuple):
    sents: list[list[str]]
    refs: list[list[int]]
    partner_refs: list[list[int]]
    partner_refs_our_view: list[list[int]]

    scenario_id: str
    chat_id: str

    real_ids: list[str]
    partner_real_ids: list[str]

    # new stuff
    text: str
    plan: list[int]
    mentions: list[int]


def get_examples(conversations, describe_plan=describe_plan_sparse):
    fields = ("dots", "text", "plan", "mentions", "outtext")
    examples = {
        key: [] for key in fields
        #for key in Conversation._fields + fields
    }
    for conversation in conversations:
        num_turns = len(conversation.sents)
        for turn in range(num_turns):
            # dont use Conversation as examples, want something that can be
            # directly fed to BART/RoBERTa
            """
            new_conversation = conversation._replace(
                sents = conversation.sents[:turn],
                refs = conversation.refs[:turn],
                partner_refs = conversation.partner_refs[:turn],
                partner_refs_our_view = conversation.partner_refs_our_view[:turn],
            )
            for key in Conversation._fields:
                examples[key].append(getattr(new_conversation, key))
            """

            is_you = conversation.sents[turn][0] == "YOU:"
            if is_you:
                # textify all dot properties
                examples["dots"].append(describe_dots(conversation.dots))
                # concatenate all text
                examples["text"].append(
                    " ".join([x for xs in conversation.sents[:turn] for x in xs])
                )
                examples["outtext"].append(" ".join(conversation.sents[turn]))
                # mention = (start idx, end idx, utterance end idx, *binary ind for 7 dots)
                raw_mentions = np.array(conversation.refs[turn]).reshape((-1, 10))[:,3:]
                raw_plan = raw_mentions.any(0).astype(int)
                examples["plan"].append(describe_plan(raw_plan))
                examples["mentions"].append(
                    " [SEP] ".join([describe_plan(m) for m in raw_mentions])
                )

    # Check number of examples for each field
    for key1 in fields:
        for key2 in fields:
            assert len(examples[key1]) == len(examples[key2])
    return examples

print(Conversation._fields)

for split in splits:
    conversations = get_conversations(split)
    examples = get_examples(conversations)

    idx = 11
    print(f"Data example in {split}")
    print("dots", examples["dots"][idx])
    print("text", examples["text"][idx])
    print("plan", examples["plan"][idx])
    print("mentions", examples["mentions"][idx])
    print("outtext", examples["outtext"][idx])

    """
    num_examples = len(examples["mentions"])
    mention_examples = {}
    mention_examples["input"] = [
        f"{dots} [MSEP] {text} [MSEP] {plan}"
        for dots, text, plan in zip(examples["dots"], examples["text"], examples["plan"])
    ]
    mention_examples["label"] = examples["mentions"]
    mention_dataset = Dataset.from_dict(mention_examples)
    mention_dataset.save_to_disk(f"hf_datasets/{split}_mentions_given_text_plan.hf")
    """

    num_examples = len(examples["plan"])
    plan_examples = {}
    plan_examples["input"] = examples["text"]
    plan_examples["input"] = [
        f"{dots} [MSEP] {text}"
        for dots, text in zip(examples["dots"], examples["text"])
    ]

    max_length = max([len(x.split()) for x in plan_examples["input"]])
    print(f"Max length of example input: {max_length}")

    plan_examples["label"] = examples["plan"]
    plan_dataset = Dataset.from_dict(plan_examples)

    m = {True: "y", False: "n"}
    feature_string = f"p{m[use_properties]}_2p{m[use_pairwise_features]}_2pu{m[use_unordered_pairwise]}_e{m[use_extrema_features]}"
    plan_dataset.save_to_disk(f"hf_datasets/{split}_plan_given_text_{feature_string}.hf")

