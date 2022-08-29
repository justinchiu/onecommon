from typing import NamedTuple

from pathlib import Path
import numpy as np


from datasets import Dataset, load_dataset

from domain import get_domain

from corpora.data import Dictionary, get_tag
from corpora.reference_sentence import ReferenceSentenceCorpus

domain = get_domain("one_common")
data = 'data/onecommon'
fold_num = 1
freq_cutoff = 20

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
    #stops = [word_dict.get_idx(w) for w in ['YOU:', 'THEM:']]
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
    sents: list[list[str]]
    refs: list[list[int]]
    partner_refs: list[list[int]]
    partner_refs_our_view: list[list[int]]

    scenario_id: str
    chat_id: str

    real_ids: list[str]
    partner_real_ids: list[str]

conversations = []
with train_path.open() as f:
    for line in f:
        line = line.strip()

        tokens = line.split()
        input_vals = [float(val) for val in get_tag(tokens, 'input')]
        words = get_tag(tokens, 'dialogue')
        word_idxs = word_dict.w2i(words)
        referent_idxs = [int(val) for val in get_tag(tokens, 'referents')]
        partner_referent_idxs = [int(val) for val in get_tag(tokens, 'partner_referents')]
        partner_referent_our_view_idxs = [int(val) for val in get_tag(tokens, 'partner_referents_our_view')]
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

            scenario_id = scenario_id,
            chat_id = chat_id,

            real_ids = real_ids,
            partner_real_ids = partner_real_ids,
        ))

"""
Process conversations into examples.
Each conversation will be turned into num_turns examples (or num_turns / 2).
Each example will look like:
    * All previous turns: each turn is a list of tokens (strings)
    * All previous refs, partner refs, partner_refs_our_view
    * scenario, chat, and ids
Eg conversation prefixes
"""

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

examples = {key: [] for key in Conversation._fields + ("text")}
for conversation in conversations:
    num_turns = len(conversation.sents)
    for turn in range(num_turns+1):
        new_conversation = conversation._replace(
            sents = conversation.sents[:turn],
            refs = conversation.refs[:turn],
            partner_refs = conversation.partner_refs[:turn],
            partner_refs_our_view = conversation.partner_refs_our_view[:turn],
        )
        for key in Conversation._fields:
            examples[key].append(getattr(new_conversation, key))

print(Conversation._fields)
import pdb; pdb.set_trace()
