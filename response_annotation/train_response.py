import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer,
    TrainingArguments,
)

from response import ResponseDb

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
    num_train_epochs=3,              # total number of training epochs
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
texts = [x[-2] for x in data]
labels = [x[-1] for x in data]
tokenized_texts = tokenizer(texts,truncation=True,padding=True)
pred_dataset = SimpleDataset(tokenized_texts, labels)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=pred_dataset,          # training dataset
    eval_dataset=pred_dataset,           # evaluation dataset
)

trainer.train()

predictions = trainer.predict(pred_dataset)

preds = predictions.predictions.argmax(-1)
labels = pd.Series(preds).map(model.config.id2label)
scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

df = pd.DataFrame(list(zip(texts,preds,labels,scores)), columns=['text','pred','label','score'])
print(df.head())
