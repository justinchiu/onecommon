import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter

from nltk import word_tokenize, pos_tag
from nltk.parse import CoreNLPParser, CoreNLPDependencyParser
import pdb
import numpy as np
from tqdm import tqdm

dialogue_tokens = []

vocab = Counter()

corenlp_parser = CoreNLPParser(url='http://localhost:9000')

def is_annotatable_markable(markable):
    if markable["generic"] or markable["no-referent"] or markable["all-referents"] or markable["anaphora"] or markable["cataphora"] or markable["predicative"]:
        return False
    else:
        return True

class Tags:
    @classmethod
    def Input(x):
        return ["<input>"] + x + ["</input>"]
    Context = "input"
    Dialogue = "dialogue"
    Output = "output"
    PartnerContext = "partner_input"

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def traverse(t):
    try:
        t.label()
    except AttributeError:
          return
    else:
        if t.label() == 'NP':
            print('NP:'+str(t.leaves()))
            print('NPhead:'+str(t.leaves()[-1]))
            for child in t:
                 traverse(child)
        else:
            for child in t:
                traverse(child)

#def extractNounPhrase(tree):


def normalize_val(val, base_val, val_range):
    normalized_val = (val - base_val) / (val_range / 2)
    assert normalized_val >= -1 and normalized_val <= 1
    return normalized_val

def create_input(kb, args):
    input_tokens = []
    xs = []
    ys = []
    sizes = []
    colors = []
    for obj in kb:
        colors.append(int(re.search(r"[\d]+", obj['color']).group(0)))
        xs.append(obj['x'])
        ys.append(obj['y'])
        sizes.append(obj['size'])
        if args.normalize:
            colors[-1] = normalize_val(colors[-1], args.base_color, args.color_range)
            xs[-1] = normalize_val(xs[-1], args.svg_radius + args.margin, args.svg_radius * 2)
            ys[-1] = normalize_val(ys[-1], args.svg_radius + args.margin, args.svg_radius * 2)
            sizes[-1] = normalize_val(sizes[-1], args.base_size, args.size_range)
        input_tokens += [str(xs[-1]), str(ys[-1]), str(sizes[-1]), str(colors[-1])]
    return input_tokens

def create_dialogue(text, agent):
    global vocab
    global corenlp_parser
    global dialogue_tokens

    dialogue_tokens = []

    for utterance in text.split("\n"):
        if utterance.startswith("{}: ".format(agent)):
            utterance_tokens = ["YOU:"]
        else:
            utterance_tokens = ["THEM:"]

        utterance_string = utterance[3:]

        word_toks = word_tokenize(utterance_string.lower())

        vocab.update(word_toks)

        utterance_tokens += word_toks

        utterance_tokens.append("<eos>")

        dialogue_tokens += utterance_tokens

    # remove last <eos>
    dialogue_tokens = dialogue_tokens[:-1] + ['<selection>']

    return ['<dialogue>'] + dialogue_tokens + ['</dialogue>']

def create_disagreements(markables, referent_annotation, agent, f1_threshold=1.0):
    disag_for_own_refs = []
    disag_for_partner_refs = []

    for markable in markables:
        markable_id = markable["markable_id"]
        if markable_id in referent_annotation:
            is_self = markable["speaker"] == agent
            ref_annotation = referent_annotation[markable_id]
            if "unidentifiable" in referent_annotation[markable_id] and referent_annotation[markable_id]["unidentifiable"]:
                continue
            if "avg_pairwise_f1" in ref_annotation:
                disagreed = ref_annotation["avg_pairwise_f1"] < f1_threshold
            else:
                disagreed = False
            if is_self:
                disag_for_own_refs.append(1 if disagreed else 0)
            else:
                disag_for_partner_refs.append(1 if disagreed else 0)
    return ['<referent_disagreements>'] + list(map(str, disag_for_own_refs)) + ['</referent_disagreements>'] +\
            ['<partner_referent_disagreements>'] + list(map(str, disag_for_partner_refs)) + ['</partner_referent_disagreements>']

def create_referents(text, markables, referent_annotation, kb_by_agent, agent):

    referent_tokens = []
    partner_referent_tokens = []
    partner_referent_our_view_tokens = []

    # map: tokens in output dialogue -> starting position in text
    token2start = []
    text_start_pos = 0

    for utterance in text.split("\n"): 
        utterance_start_pos = 0
        token2start.append(text_start_pos + utterance_start_pos)
        utterance_start_pos += len("0: ")
        utterance_string = utterance[utterance_start_pos:]

        word_toks = word_tokenize(utterance_string.lower())

        for word_tok in word_toks:
            token2start.append(text_start_pos + utterance_start_pos)
            utterance_start_pos += len(word_tok)
            while (text_start_pos + utterance_start_pos) < len(text) and text[text_start_pos + utterance_start_pos] == " ":
                utterance_start_pos += 1

        token2start.append(text_start_pos + utterance_start_pos)
        text_start_pos += len(utterance) + 1

    #for i in range(len(token2start) - 1):
    #    print(token2start[i])
    #    print(text[token2start[i]:token2start[i+1]])

    assert len(token2start) == len(dialogue_tokens)


    for markable in markables:
        markable_id = markable["markable_id"]
        if markable_id in referent_annotation:
            is_self = markable["speaker"] == agent
            if "unidentifiable" in referent_annotation[markable_id] and referent_annotation[markable_id]["unidentifiable"]:
                continue
            markable_id = markable["markable_id"]
            start = markable["start"]
            end = markable["end"]

            for i in range(len(token2start)):
                if i == len(token2start) - 1 or start <= token2start[i]:
                    start_tok = i
                    break

            for i in range(len(token2start)):
                if i == len(token2start) - 1 or end <= token2start[i+1]:
                    end_tok = i
                    break

            for i in range(len(token2start)):
                if start <= token2start[i] and dialogue_tokens[i] in ["<eos>", "<selection>"]:
                    end_of_utterance_tok = i
                    break

            # fix mistake due to tokenization mistaket
            if start_tok > end_tok:
                if chat_id == "C_b07ee61970504668a7b63c3973936129" and markable_id == "M4":
                    start_tok = 29
                    end_tok = 29
                elif chat_id == "C_22b80a8d7a6e417da0de423aa9bad760":
                    start_tok = 15
                    end_tok = 15

            # use end of dialogue token
            #end_of_utterance_tok = len(dialogue_tokens) - 1

            #print(text[start:end])
            #print(dialogue_tokens[start_tok:end_tok+1])
            #print(dialogue_tokens[end_of_utterance_tok])

            if is_self:
                all_destination_toks = [(referent_tokens, markable['speaker'])]
            else:
                all_destination_toks = [(partner_referent_tokens, markable['speaker']), (partner_referent_our_view_tokens, 1 - markable['speaker'])]

            for destination_toks, agent_view in all_destination_toks:
                destination_toks.append(str(start_tok))
                destination_toks.append(str(end_tok))
                destination_toks.append(str(end_of_utterance_tok))

                for ent in kb_by_agent[agent_view]:
                    if "agent_{}_{}".format(markable['speaker'], ent['id']) in referent_annotation[markable_id]["referents"]:
                        destination_toks.append("1")
                    else:
                        destination_toks.append("0")

    return ['<referents>'] + referent_tokens + ['</referents>'] +\
                ['<partner_referents>'] + partner_referent_tokens + ['</partner_referents>'] + \
                ['<partner_referents_our_view>'] + partner_referent_our_view_tokens + ['</partner_referents_our_view>']


def create_output(kb, events, agent):
    ids = []
    for obj in kb:
        ids.append(obj['id'])
    select_id = None
    for event in events:
        if event['action'] == 'select' and agent == event['agent']:
            select_id = ids.index(event['data'])
    return ['<output>', str(select_id), '</output>']

def create_real_ids(kb):
    real_ids = []
    for obj in kb:
        real_ids.append(obj['id'])
    return real_ids

def create_partner_real_ids(kb):
    real_ids = []
    for obj in kb:
        real_ids.append(obj['id'])
    return ['<partner_real_ids>'] + real_ids + ['</partner_real_ids>']

def create_scenario_id(scenario_id):
    return ['<scenario_id>'] + [scenario_id] + ['</scenario_id>']

def create_agent(agent):
    return ['<agent>'] + [str(agent)] + ['</agent>']

def create_chat_id(chat_id):
    return ['<chat_id>'] + [chat_id] + ['</chat_id>']

def create_markables(markables):
    markable_tokens = []
    for markable in markables:
        markable_id = markable["markable_id"]
        markable_text = markable["text"]
        markable_tokens.append(markable_id)
        markable_tokens.append(markable_text)
    return ['<markables>'] + markable_tokens + ['</markables>']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_range', type=int, default=150, help='range of color')
    parser.add_argument('--size_range', type=int, default=6, help='range of size')
    parser.add_argument('--base_size', type=int, default=10, help='base of size')
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--valid_proportion', type=float, default=0.1)
    parser.add_argument('--test_proportion', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1, help='range of size')
    args = parser.parse_args()

    # current support
    args.base_color = 128
    args.svg_radius = 200
    args.margin = 15
    
    np.random.seed(args.seed)

    args.train_proportion = 1 - args.valid_proportion - args.test_proportion

        # contains all dialogues (6.7K)
    dialogue_corpus = read_json("final_transcripts.json")

    # save out scenarios
    with open("../experiments/data/onecommon/static/scenarios.json", "w") as f:
        json.dump([x["scenario"] for x in dialogue_corpus], f)

        # contains successful dialogues (5.1K)
    markable_annotation = read_json("markable_annotation.json")
        # contains successful dialogues (5.1K)
    aggregated_referent_annotation = read_json("aggregated_referent_annotation.json")
    
    chat_ids = list(aggregated_referent_annotation.keys())

    # shuffle corpus
    np.random.shuffle(chat_ids)

    total_size = len(chat_ids)
    split_index = [0, int(total_size * args.train_proportion),
                    int(total_size * (args.train_proportion + args.valid_proportion)), -1]
    tags = Tags()

    # log 
    output_file_names = [
        'train_context_' + str(args.seed),
        'valid_context_' + str(args.seed),
        'test_context_' + str(args.seed),
    ]

    for i, output_file in enumerate(output_file_names):
        with open('{}.txt'.format('../experiments/data/onecommon/static/' + output_file), 'w') as out:
            start = split_index[i]
            end = split_index[i+1]
            for chat_id in tqdm(chat_ids[start:end]):    
                chat = [chat for chat in dialogue_corpus if chat['uuid'] == chat_id]
                chat = chat[0]
                scenario = chat["scenario"]

                out.write(scenario['uuid'] + "\n")
                for agent in [0, 1]:
                    out.write(" ".join(create_input(scenario['kbs'][agent], args)) + "\n")
                for agent in [0, 1]:
                    out.write(" ".join(create_real_ids(scenario['kbs'][agent])) + "\n")

