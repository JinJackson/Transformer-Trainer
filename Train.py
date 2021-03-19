import dataset
from dataset import IMDbDataset
from parser1 import args

from model.BertClassification import BertClassificationModel
from transformers import TrainingArguments, AdamW, get_linear_schedule_with_warmup, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch.utils.data import DataLoader
import os
import torch

import time

print('Processing Data...')
start_time = time.time()
train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels = dataset.get_data()

train_dataset = IMDbDataset(encodings=train_encodings, labels=train_labels) #20000条
eval_dataset = IMDbDataset(encodings=val_encodings, labels=val_labels)  #5000条
test_dataset = IMDbDataset(encodings=test_encodings, labels=test_labels)  # 25000条
print('End processing data, costing time:' + str(time.time() - start_time))

start_time = time.time()

total_steps = len(train_dataset) * args.epochs

model = BertClassificationModel.from_pretrained(args.config)

print('Loading Model...costing time:' + str(time.time()-start_time))


# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

def train(model):
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # optimizers 默认为AdamW, 并且自带get_linear_schedule_with_warmup()，参数由Training_args控制
        warmup_steps=int(args.warmup_steps * total_steps),
        weight_decay=args.decay_weight,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.grad_accumulate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #compute_metrics=compute_metrics
        # 也可以更换optimizer
        #optimizers=(Optimizer, scheduler)
    )

    trainer.train()

    trainer.evaluate()


start_time = time.time()
train(model)
print('Ending Training, cost total time:' + str(time.time() - start_time))