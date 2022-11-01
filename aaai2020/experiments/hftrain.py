from argparse import ArgumentParser
from rich.progress import track
import json
from pathlib import Path

from itertools import zip_longest

import torch
import torch.nn as nn
import numpy as np

from datasets import Dataset, load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import TrainingArguments, Trainer
from transformers.models.bart.modeling_bart import shift_tokens_right

from models.dotbart import DotBartForConditionalGeneration

import hfutils

def get_datasets(args, dataset, model, tokenizer, do_eval=False):
    train = Dataset.load_from_disk(f"hf_datasets/train_{dataset}.hf")
    valid = Dataset.load_from_disk(f"hf_datasets/valid_{dataset}.hf")
    test = Dataset.load_from_disk(f"hf_datasets/test_{dataset}.hf")

    use_raw_dots = "rd" in dataset

    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['input'],
            max_length = args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        target_encodings = tokenizer.batch_encode_plus(
            example_batch['label'],
            max_length = args.output_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        
        labels = target_encodings['input_ids']
        decoder_input_ids = shift_tokens_right(
            torch.tensor(labels),
            model.config.pad_token_id,
            model.config.eos_token_id,
        ).numpy()
        labels[labels[:, :] == model.config.pad_token_id] = -100

        attention_mask = input_encodings['attention_mask']
        if use_raw_dots:
            bsz, time = attention_mask.shape
            attention_mask = np.concatenate((
                np.ones((bsz, 7), dtype=attention_mask.dtype),
                attention_mask,
            ), 1)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            "scenario_ids": example_batch["scenario_id"],
            "chat_ids": example_batch["chat_id"],
            "agents": example_batch["agent"],
        }
        if "dots" in example_batch:
            encodings["dots"] = np.array(example_batch["dots"])

        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = [
            'input_ids',
            'labels',
            'decoder_input_ids',
            'attention_mask',
        ]
        if use_raw_dots:
            columns.append("dots")
        dataset.set_format(type='torch', columns=columns, output_all_columns=do_eval)
        return dataset

    tokenized_train = process_dataset(train)
    tokenized_valid = process_dataset(valid)
    tokenized_test = process_dataset(test)

    return (tokenized_train, tokenized_valid, tokenized_test)


def train(args):
    dataset = args.dataset
    tokenizer = hfutils.get_bart_tokenizer()

    # model
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", forced_bos_token_id=0,
    )
    model.resize_token_embeddings(len(tokenizer))

    # data
    tokenized_train, tokenized_valid, tokenized_test = get_datasets(
        args, dataset, model, tokenizer, do_eval=False)

    # process raw dots
    use_raw_dots = "rd" in dataset
    if use_raw_dots:
        # raw dots are 7 dots x 4 dims (x, y, size, color)
        dot_encoder = nn.Linear(4, model.get_input_embeddings().embedding_dim)
        model2 = DotBartForConditionalGeneration(model.config, dot_encoder)
        model2.model = model.model
        model2.lm_head = model.lm_head
        model2.final_logits_bias = model.final_logits_bias
        # replace original model
        model = model2

    training_args = TrainingArguments(
        output_dir=f"./hf-results-{args.dataset}-l{args.learning_rate}-b{args.batch_size}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy = "steps",
        logging_dir=f"./hf-log-{args.dataset}-l{args.learning_rate}-b{args.batch_size}",

        # supplied args
        learning_rate = args.learning_rate,

        # num steps for stuff
        eval_steps = 500,
        save_steps = 1000,
        save_total_limit = 3,
        metric_for_best_model = "eval_loss",
        load_best_model_at_end = True,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_valid,
    )
    trainer.train()

    savedir = f"./hf-save-{args.dataset}-l{args.learning_rate}-b{args.batch_size}"
    tokenizer.save_pretrained(savedir)
    model.save_pretrained(savedir)

@torch.inference_mode()
def evaluate(args):
    dataset = args.dataset
    use_raw_dots = "rd" in dataset
    IS_TEXT = args.dataset[:4] == "text"

    # forgot to save tokenizer and model, rerun training and fix this
    tokenizer = hfutils.get_bart_tokenizer()

    # model
    checkpoint_string = f"checkpoint-{args.checkpoint}"

    output_dir=f"./hf-results-{args.dataset}-l{args.learning_rate}-b{args.batch_size}/{checkpoint_string}"
    model_class = (DotBartForConditionalGeneration
        if use_raw_dots else BartForConditionalGeneration)
    # TODO: initialize dot_encoder inside model to make loading easier
    dot_encoder = nn.Linear(4, 1024)
    if use_raw_dots:
        model = model_class.from_pretrained(
            output_dir, dot_encoder = dot_encoder, forced_bos_token_id=0,
        )
    else:
        model = model_class.from_pretrained(
            output_dir, forced_bos_token_id=0,
        )
    model.resize_token_embeddings(len(tokenizer))
    
    generation_path = Path(
        f"./hf-generations-{args.eval_dataset}-l{args.learning_rate}-"
        f"b{args.batch_size}/{checkpoint_string}.gen.json"
    )
    print(f"Saving generations to {str(generation_path)}")
    generation_path.parent.mkdir(parents=True,exist_ok=True)

    # data
    tokenized_train, tokenized_valid, tokenized_test = get_datasets(
        args, dataset, model, tokenizer, do_eval=True)

    ids_inputs_labels_outputs = []

    exact_match = 0
    num_examples = 0
    bsz = args.eval_batch_size
    max_examples = len(tokenized_valid)
    #for batch_idx in track(range(max_examples // bsz)):
    for batch_idx in range(max_examples // bsz):
        batch = tokenized_valid[batch_idx * bsz: (batch_idx+1) * bsz]
        # one at a time
        # prepare model input
        input_ids = batch["input_ids"]
        dots = None
        if "dots" in batch:
            dots = batch["dots"].view(-1, 7 ,4)

        emb = model.get_input_embeddings()
        enc = model.get_encoder()

        token_embs = emb(input_ids)
        inputs_embeds = token_embs
        if use_raw_dots:
            dot_embs = model.dot_encoder(dots)
            inputs_embeds = torch.cat([dot_embs, token_embs], 1)
            #encoder_outputs = enc(inputs_embeds = inputs_embeds)
        inputs_embeds = inputs_embeds * enc.embed_scale

        output = model.generate(
            #encoder_outputs = encoder_outputs,
            inputs_embeds = inputs_embeds,
            #input_ids = input_ids,
            #dots = dots,
            num_beams = args.beam_size if IS_TEXT else None,
            num_return_sequences = args.beam_size if IS_TEXT else None,
            output_scores = True,
            return_dict_in_generate = True,
            max_new_tokens = 80,
        )
        output_dots = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        labels = batch["labels"]
        if "plan_given" in args.dataset:
            for i, output_dot in enumerate(output_dots):
                label = labels[i][labels[i] != -100]
                label_dots = tokenizer.decode(label, skip_special_tokens=True)

                output_set = set(output_dot.replace(",", "").split())
                label_set = set(label_dots.replace(",", "").split())

                if output_set == label_set:
                    exact_match += 1
                num_examples += 1
        elif "mentions_given" in args.dataset:
            for i, output_dot in enumerate(output_dots):
                # iterate over batches
                label = labels[i][labels[i] != -100]
                label_dots = tokenizer.decode(label, skip_special_tokens=True)
                #print(output_dot)
                #print(label_dots)

                split_output_dots = output_dot.split(" [SEP] ")
                split_label_dots = label_dots.split(" [SEP] ")
                for j, label_mention in enumerate(split_label_dots):
                    if j < len(split_output_dots):
                        output_mention = split_output_dots[j]

                        # each turn is a sequence of mentions,
                        # so we want to evaluate each mention as a set
                        output_set = set(output_mention.replace(",", "").split())
                        label_set = set(label_mention.replace(",", "").split())

                        if output_set == label_set:
                            exact_match += 1
                    num_examples += 1

        if args.dataset[:4] != "text":
            print(f"Exact match @ batch {batch_idx}: {exact_match} / {num_examples}")
        else:
            # text generation
            inputs = tokenizer.batch_decode(model_input, skip_special_tokens=True)
            outputs = output_dots # flattened bsz * num_return_sequences
            scores = output.sequences_scores # or output.scores?
            for i in range(bsz):
                chat_id = batch["chat_ids"][i]
                scenario_id = batch["scenario_ids"][i]
                agent = batch["agents"][i]
                input_str = inputs[i]
                output_strs = outputs[i*args.beam_size:(i+1)*args.beam_size]
                label = labels[i][labels[i] != -100]
                label_str = tokenizer.decode(label, skip_special_tokens=True)
                ids_inputs_labels_outputs.append(
                    (chat_id, scenario_id, agent, input_str, label_str, output_strs)
                )

            if args.num_batches > 0 and (batch_idx+1) % args.num_batches == 0:
                with generation_path.open("w") as f:
                    json.dump(ids_inputs_labels_outputs, f)


    print(f"Exact match: {exact_match} / {num_examples}")
    if IS_TEXT:
        with generation_path.open("w") as f:
            json.dump(ids_inputs_labels_outputs, f)



if __name__ == "__main__":
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-5,
        help='learning rate',
    )
    parser.add_argument(
        '--max_length', type=int, default=512,
        help='input example max length',
    )
    parser.add_argument(
        '--output_max_length', type=int, default=128,
        help='output example max length',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='batch size',
    )
    parser.add_argument(
        '--epochs', type=int, default=7,
        help='epochs',
    )
    parser.add_argument(
        '--dataset',
        choices = [
            #"plan_given_text",
            #"mentions_given_text_plan",
            "plan_given_text_py_2py_2puy_en",
            "text_given_plan_py_2py_2puy_en_sdn",
            "text_given_plan_py_2py_2puy_en_sdy",
            "text_given_plan_planspecific",

            "text_given_plan_py_2py_2puy_en_sdn_psn_uy",
            "text_given_plan_py_2py_2puy_en_sdy_psn_uy",
            "text_given_plan_py_2py_2puy_en_sdy_psy_uy",

            "text_given_plan_py_2py_2puy_en_sdn_psn_un",
            "text_given_plan_py_2py_2puy_en_sdy_psn_un",
            "text_given_plan_py_2py_2puy_en_sdy_psy_un",

            # 10/4
            "text_given_plan_SI_CO_RX_RY_RS_RC_ur_sd_ps_c_sl_s",

            # 10/5
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgt_ur_sd_ps__c_sl_s_mps7_dh",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgt_ur_sd_ps_sr_c_sl_s_mps5_dh",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgt_ur_sd_ps_sr_c_sl_s_mps5_",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgt__sd_ps_sr_c_sl_s_mps5_dh",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_c_sl_s_mps5_dh",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_c_sl_s_mps5_dh",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_c_sl_s_mps5_",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_c_sl_s_mps5_",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_c_sl_s_mps7_",
            "text_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_c_sl_s_mps7_",

            # 10/9
            "text_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_c_sl_s_mps7_",
            "text_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_c_sl_s_mps7_",
            # 10/11
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgt__sd_ps_sr_c_sl_s_mps5_",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_c_sl_s_mps7_",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_c_sl_s_mps7_",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_c_sl_s_mps5_",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_c_sl_s_mps5_",
            # 10/19
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_c_sl_s_mps5___",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_c_sl_s_mps5__ma_",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_c_sl_s_mps5__ma_b",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelTgts__sd_ps_sr_cd_c_sl_s_mps5__ma_b",
            # 10/19 
            "textmention_given_mention_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd__c_sl_s_mps25__ma_",
            "textmention_given_mention_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_mps25__ma_",
            # 10/24 coref
            "textmention_given_mention_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd__c_sl_s_co_mps25__ma_",
            "textmention_given_mention_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25__ma_",
            "textmention_given_mention_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25__ma_b",

            # 10/27 mention prediction
            "mentions_given_text_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh_lt_ma_",
            "mentions_given_textmention_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh_lt_ma_",
            "mentions_given_text_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh__ma_",
            "mentions_given_textmention_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh__ma_",
            "mentions_given_text_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh_lt_ma_",
            "mentions_given_textmention_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh_lt_ma_",
            "mentions_given_text_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh__ma_",
            "mentions_given_textmention_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh__ma_",

            # 10/31 mention prediction
            "mentions_given_text_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh_lt_ma__rd",
            "mentions_given_textmention_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh_lt_ma__rd",
            "mentions_given_text_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh__ma__rd",
            "mentions_given_textmention_plan_consel_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_ms_c_sl_s_co_mps25_dh__ma__rd",
        ],
        default = "plan_given_text_planspecific",
        help="Dataset",
    )

    parser.add_argument(
        "--eval_dataset",
        default = None,
        help = "Evaluation dataset. In particular, one with metadata"
        "interventions. If None, defaults to args.dataset.",
    )

    parser.add_argument(
        '--eval', action = "store_true",
        help="Perform model evaluation on latest checkpoint",
    )
    parser.add_argument(
        '--checkpoint', type = str, default = "32000",
        help="Checkpoint iteration number for eval model loading",
    )
    parser.add_argument(
        '--beam_size', type = int, default = 4,
        help="Beam size for generation eval",
    )
    parser.add_argument(
        '--num_batches', type = int, default = 5,
        help="Save every num_batches in generation eval",
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=16,
        help='Evaluation batch size',
    )

    args = parser.parse_args()

    if args.eval_dataset is None:
        args.eval_dataset = args.dataset

    if not args.eval:
        train(args)
    else:
        evaluate(args)
