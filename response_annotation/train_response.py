# IMPORTANT: data looks like: 'Yes I have it'

import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer,
    TrainingArguments,
)

from datasets import load_metric

from response import ResponseDb

import random

# constants
SENTIMENT = "sentiment"
NLI = "nli"

datatype = SENTIMENT
#datatype = NLI

# seeds
seed = 1111
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=15,             # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model_name = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# ignore_mismatched_sizes replaces head with a different one, must be trained
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3, ignore_mismatched_sizes=True,
)

db = ResponseDb()
data = db.get_all()
texts = None
if datatype == SENTIMENT:
    texts = [x[-2] for x in data]
elif datatype == NLI:
    texts = [f"{x[-3]} </s></s> {x[-2]}" for x in data]
labels = [x[-1] for x in data]
tokenized_texts = tokenizer(texts,truncation=True,padding=True)

N = len(texts)

train_tokenized_texts = tokenizer(texts[:N - N//4], truncation=True, padding=True)
valid_tokenized_texts = tokenizer(texts[N-N//4:], truncation=True, padding=True)
train_labels = labels[:N - N//4]
valid_labels = labels[N-N//4:]

train_dataset = SimpleDataset(train_tokenized_texts, train_labels)
valid_dataset = SimpleDataset(valid_tokenized_texts, valid_labels)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
)

trainer.train()

predictions = trainer.predict(valid_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

metric = load_metric("accuracy")
accuracy = metric.compute(predictions=preds, references=predictions.label_ids)
print(accuracy)

save_directory = "./save_pretrained"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
#trainer.save_model(f"response-model-{datatype}-{accuracy:.2f}")
