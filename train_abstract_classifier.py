### train

import warnings
warnings.simplefilter('ignore')

import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import evaluate
from transformers import RobertaTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset
import logging
logging.basicConfig(level=logging.ERROR)
from huggingface_hub import login

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

id2label = {0: "NO", 1: "YES"}
label2id = {"NO": 0, "YES": 1}

def preprocess(data, tokenizer):
    return tokenizer(data["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {"accuracy":acc["accuracy"], "f1":f1["f1"]}
    
def format_data(data):
    data = {"label":data["label"], "text":data["text"]}
    data = Dataset.from_dict(data)
    return data

def main():
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    login(token=pd.read_csv(cfg["huggingface_token"]).columns[0])
    df = pd.read_csv(cfg["train_data"])
    df = df.sample(frac=1)
    data = format_data(df)
    tokenizer = RobertaTokenizer.from_pretrained(cfg["pretrained_model_name"], padding=True, truncation=True, do_lower_case=True)
    tokenized_data = data.map(preprocess, batched=True, fn_kwargs={"tokenizer":tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    folds = StratifiedKFold(n_splits=cfg["cv_num_splits"], shuffle=True)
    splits = folds.split(np.zeros(tokenized_data.num_rows), tokenized_data["label"])
    dataset_list = []
    for train_i, val_i in splits:
        temp_train = tokenized_data.select(train_i)
        temp_val = tokenized_data.select(val_i)
        dataset_list.append((temp_train, temp_val))

    accuracies = []
    f1s = []
    for i in range(cfg["cv_num_splits"]):
        model = AutoModelForSequenceClassification.from_pretrained(cfg["pretrained_model_name"], num_labels=2, id2label=id2label, label2id=label2id)
        training_args = TrainingArguments(
            output_dir=f"cache/{cfg['model_save_name']}_{i}",
            learning_rate=cfg["train_learning_rate"],
            weight_decay=cfg["train_weight_decay"],
            per_device_train_batch_size=cfg["train_batch_size"],
            per_device_eval_batch_size=16,
            num_train_epochs=cfg["train_epochs"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to="none",
            logging_steps=1,
            save_total_limit=1,
            push_to_hub=True,
            disable_tqdm=False
        )
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
        
    print(f"Accuracies: {accuracies}")
    print(f"f1s: {f1s}")
    print(f"Average accuracy over 5-fold CV: {np.mean(accuracies)}")
    print(f"Average f1 score over 5-fold CV: {np.mean(f1s)}")
    print(f"Best model in terms of: accuracy - {np.argmax(accuracies)}, f1 - {np.argmax(f1s)}")

main()