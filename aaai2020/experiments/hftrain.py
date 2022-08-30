from argparse import ArgumentParser

from datasets import Dataset, load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import TrainingArguments, Trainer
from transformers.models.bart.modeling_bart import shift_tokens_right


def main(args):
    # data
    dataset = args.dataset
    train = Dataset.load_from_disk(f"hf_datasets/train_{dataset}.hf")
    valid = Dataset.load_from_disk(f"hf_datasets/valid_{dataset}.hf")
    test = Dataset.load_from_disk(f"hf_datasets/test_{dataset}.hf")

    # model
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large", forced_bos_token_id=0,
    )
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['input'],
            pad_to_max_length=True,
            max_length=1024,
            truncation=True,
        )
        target_encodings = tokenizer.batch_encode_plus(
            example_batch['label'],
            pad_to_max_length=True,
            max_length=1024,
            truncation=True,
        )
        
        labels = target_encodings['input_ids']
        decoder_input_ids = shift_tokens_right(
            labels,
            model.config.pad_token_id,
            model.config.eos_token_id,
        )
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
        output_dir="./hfresults",
        num_train_epochs=15,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        #evaluation_strategy="epoch",
        logging_dir="./hflog",
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_valid,
    )
    train.train()


if __name__ == "__main__":
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-5,
        help='learning rate',
    )
    parser.add_argument(
        '--dataset', choices=["plan_given_text", "mentions_given_text_plan"],
        default = "plan_given_text",
        help="Dataset",
    )
    args = parser.parse_args()

    main(args)
