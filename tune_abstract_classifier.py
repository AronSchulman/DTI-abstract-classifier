### tune

import warnings
warnings.simplefilter('ignore')
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ray import tune
import evaluate
from transformers import RobertaTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset
import logging
logging.basicConfig(level=logging.ERROR)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

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

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(cfg["pretrained_model_name"], return_dict=True)

def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(float(cfg["learning_rate"][0]), float(cfg["learning_rate"][1])),
        "weight_decay" : tune.loguniform(float(cfg["weight_decay"][0]), float(cfg["weight_decay"][1])),
        "per_device_train_batch_size": tune.choice(cfg["batch_size"]),
    }
    
def main():
    
    id2label = {0: "NO", 1: "YES"}
    label2id = {"NO": 0, "YES": 1}
    
    df = pd.read_csv(cfg["train_data"])
    df = df.sample(frac=1, random_state=cfg["seed"])
    data = format_data(df)
    tokenizer = RobertaTokenizer.from_pretrained(cfg["pretrained_model_name"], padding=True, truncation=True, do_lower_case=True)
    tokenized_data = data.map(preprocess, batched=True, fn_kwargs={"tokenizer":tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    folds = StratifiedKFold(n_splits=cfg["cv_num_splits"], random_state=cfg["seed"], shuffle=True)
    splits = folds.split(np.zeros(tokenized_data.num_rows), tokenized_data["label"])
    dataset_list = []
    for train_i, val_i in splits:
        temp_train = tokenized_data.select(train_i)
        temp_val = tokenized_data.select(val_i)
        dataset_list.append((temp_train, temp_val))
    
    
    training_args = TrainingArguments(
        output_dir=cfg["tune_log_dir"],
        num_train_epochs=cfg["num_epochs"],
        evaluation_strategy="epoch",
        eval_steps=1,
        push_to_hub=False,
        disable_tqdm=True
    )
    trainer = Trainer(
        args=training_args,
        train_dataset=dataset_list[0][0],
        eval_dataset=dataset_list[0][1],
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_init=model_init,
        compute_metrics=compute_metrics
    )

    analysis = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=ray_hp_space,
        n_trials=cfg["num_trials"],
        storage_path=cfg["tune_log_dir"]
    )
    
    analysis.run_summary.dataframe(metric="objective", mode="max").to_csv(cfg["tune_save_dir"], index=False)
    print(f"Hyperparameter optimization complete!\nSummary dataframe saved at {cfg['tune_save_dir']}.\nBest hyperparameters: {analysis.run_summary.get_best_config(metric='objective', mode='max')}.\nUse these values for training your model.")

main()