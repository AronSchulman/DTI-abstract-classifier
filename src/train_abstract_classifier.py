import sys

import warnings
warnings.simplefilter('ignore')
import os
os.environ['HF_HOME'] = '../../huggingface_models/'

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import transformers
import evaluate
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, pipeline
from datasets import Dataset
import logging
logging.basicConfig(level=logging.ERROR)

from huggingface_hub import login
login(token=pd.read_csv("../../huggingface_models/token").columns[0])



from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

pt_model_name = "allenai/biomed_roberta_base"

id2label = {0: "NO", 1: "YES"}
label2id = {"NO": 0, "YES": 1}
model = AutoModelForSequenceClassification.from_pretrained(pt_model_name, num_labels=2, id2label=id2label, label2id=label2id)

def preprocess(data):
    return tokenizer(data["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {"accuracy":acc["accuracy"], "f1":f1["f1"]}

def format_data(data, col_to_keep):
    data = df[[col_to_keep, "text"]]
    data = data.rename(columns={col_to_keep:"label"})
    data = {"label":data["label"], "text":data["text"]}
    data = Dataset.from_dict(data)
    
    return data

def form_weight_tensor(train_data):
    num_labels = len(train_data[:]["label"])
    num_one_labels = sum(train_data[:]["label"])
    num_zero_labels = num_labels - num_one_labels
    class_weight = torch.tensor([num_labels/(2*num_zero_labels), num_labels/(2*num_one_labels)], device=device)
    return class_weight

def main():

    categories = ["Machine Learning", "Deep Learning", "Attention-based", "Docking", "Classification", "Regression", "Cross-validation", "Independent testing", "SOTA comparison", "Experimental validation", "Accuracy", "RMSE", "AUROC", "Spearman/Pearson", "F1-score", "Other"]
    train_batch_size = 8
    learning_rate = 0.00008861577452533074
    weight_decay = 0.0029210748185657135
    train_epochs = 10
    for categ in categories:
        df = pd.read_csv(f"../data/train/train_{categ[0:4]}.csv")
        df = df.sample(frac=1)
        data = format_data(df, categ)
        tokenizer = RobertaTokenizer.from_pretrained(pt_model_name, padding=True, truncation=True, do_lower_case=True)
        tokenized_data = data.map(preprocess, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        folds = StratifiedKFold(n_splits=5, shuffle=True)
        splits = folds.split(np.zeros(tokenized_data.num_rows), tokenized_data["label"])
        dataset_list = []
        for train_i, val_i in splits:
            temp_train = tokenized_data.select(train_i)
            temp_val = tokenized_data.select(val_i)
            dataset_list.append((temp_train, temp_val))
    
        accuracies = []
        f1s = []
        for i in range(5):
            model = AutoModelForSequenceClassification.from_pretrained(pt_model_name, num_labels=2, id2label=id2label, label2id=label2id)
            training_args = TrainingArguments(
                output_dir=f"{categ[0:4]}_{i}",
                learning_rate=learning_rate,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=16,
                num_train_epochs=train_epochs,
                weight_decay=weight_decay,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="f1",
                greater_is_better=True,
                load_best_model_at_end=True,
                report_to="none",
                logging_steps=1,
                push_to_hub=False
            )
            class_weights = form_weight_tensor(dataset_list[i][0])
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_list[i][0],
                eval_dataset=dataset_list[i][1],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
    
            trainer.train()
            evals = trainer.evaluate(dataset_list[i][1])
            accuracies.append(evals["eval_accuracy"])
            f1s.append(evals["eval_f1"])
        
        print(f"Category: {categ}")
        print(f"Accuracies: {accuracies}")
        print(f"f1s: {f1s}")
        print(f"Average accuracy over 5-fold CV: {np.mean(accuracies)}")
        print(f"Average f1 score over 5-fold CV: {np.mean(f1s)}")
        print(f"Best model in terms of: accuracy - {np.argmax(accuracies)}, f1 - {np.argmax(f1s)}")

main()