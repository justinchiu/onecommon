from argparse import ArgumentParser
from rich.progress import track
import json

from itertools import zip_longest

from datasets import Dataset, load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import TrainingArguments, Trainer
from transformers.models.bart.modeling_bart import shift_tokens_right

import hfutils

import torch

def train(args):
    # data
    dataset = args.dataset
    train = Dataset.load_from_disk(f"hf_datasets/train_{dataset}.hf")
    valid = Dataset.load_from_disk(f"hf_datasets/valid_{dataset}.hf")
    test = Dataset.load_from_disk(f"hf_datasets/test_{dataset}.hf")

    tokenizer = hfutils.get_bart_tokenizer()
    # model
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", forced_bos_token_id=0,
    )
    model.resize_token_embeddings(len(tokenizer))

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
        
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }

        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = ['input_ids', 'labels', 'decoder_input_ids','attention_mask',] 
        dataset.set_format(type='torch', columns=columns)
        return dataset

    tokenized_train = process_dataset(train)
    tokenized_valid = process_dataset(valid)
    tokenized_test = process_dataset(test)

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

def evaluate(args):
    # data
    dataset = args.dataset
    train = Dataset.load_from_disk(f"hf_datasets/train_{dataset}.hf")
    valid = Dataset.load_from_disk(f"hf_datasets/valid_{dataset}.hf")
    test = Dataset.load_from_disk(f"hf_datasets/test_{dataset}.hf")

    # forgot to save tokenizer and model, rerun training and fix this
    tokenizer = hfutils.get_bart_tokenizer()

    # model
    output_dir=f"./hf-results-{args.dataset}-l{args.learning_rate}-b{args.batch_size}/checkpoint-33000"
    output_dir=f"./hf-results-{args.dataset}-l{args.learning_rate}-b{args.batch_size}/checkpoint-11500"
    model = BartForConditionalGeneration.from_pretrained(
        output_dir, forced_bos_token_id=0,
    )
    model.resize_token_embeddings(len(tokenizer))

    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['input'],
            max_length = 768,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        target_encodings = tokenizer.batch_encode_plus(
            example_batch['label'],
            max_length = 128,
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
        
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }

        return encodings

    def process_dataset(dataset):
        dataset = dataset.map(convert_to_features, batched=True)
        columns = ['input_ids', 'labels', 'decoder_input_ids','attention_mask',] 
        dataset.set_format(type='torch', columns=columns)
        return dataset

    tokenized_train = process_dataset(train)
    tokenized_valid = process_dataset(valid)
    tokenized_test = process_dataset(test)

    IS_TEXT = args.dataset[:4] == "text"

    exact_match = 0
    num_examples = 0
    bsz = 4
    max_examples = len(tokenized_valid)
    #for batch_idx in track(range(max_examples // bsz)):
    for batch_idx in range(max_examples // bsz):
        batch = tokenized_valid[batch_idx * bsz: (batch_idx+1) * bsz]
        # one at a time
        model_input = batch["input_ids"]
        output = model.generate(
            model_input,
            num_beams = 16 if IS_TEXT else None,
            num_return_sequences = 16 if IS_TEXT else None,
            output_scores = True,
            return_dict_in_generate = True,
            max_new_tokens = 40,
        )
        output_dots = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        labels = batch["labels"]
        if args.dataset == "plan_given_text":
            for i, output_dot in enumerate(output_dots):
                label = labels[i][labels[i] != -100]
                label_dots = tokenizer.decode(label, skip_special_tokens=True)

                output_set = set(output_dot.replace(",", "").split())
                label_set = set(label_dots.replace(",", "").split())

                if output_set == label_set:
                    exact_match += 1
                num_examples += 1
        elif args.dataset == "mentions_given_text_plan":
            for i, output_dot in enumerate(output_dots):
                label = labels[i][labels[i] != -100]
                label_dots = tokenizer.decode(label, skip_special_tokens=True)

                # each turn is a sequence of mentions,
                # so we want to evaluate each mention as a set
                for output_mention, label_mention in zip_longest(
                    output_dot.split(" [SEP] "),
                    label_dots.split(" [SEP] "),
                    fillvalue="",
                ):
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
            outputs = output_dots
            scores = output.sequences_scores # or output.scores?
            import pdb; pdb.set_trace()

    print(f"Exact match: {exact_match} / {num_examples}")


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
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_c_sl_s_mps5_",
            "textmention_mention_given_plan_SI_CO_RX_RY_RS_RC_SrcRelsTgt__sd_ps_sr_cd_c_sl_s_mps5__ma",
        ],
        default = "plan_given_text_planspecific",
        help="Dataset",
    )
    parser.add_argument(
        '--eval', action = "store_true",
        help="Perform model evaluation on latest checkpoint",
    )

    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        evaluate(args)
