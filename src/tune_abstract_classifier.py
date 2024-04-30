import warnings
warnings.simplefilter('ignore')
import os
os.environ['HF_HOME'] = '../../huggingface_models/'
os.environ["WANDB_PROJECT"] = "expert_opinion"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # or "end"
os.environ["WANDB_CACHE_DIR"] = '../../huggingface_models/wandb_cache'

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from ray import tune
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

def format_data(df, col_to_keep):
    data = df[[col_to_keep, "text"]]
    data = data.rename(columns={col_to_keep:"label"})
    data = {"label":data["label"], "text":data["text"]}
    data = Dataset.from_dict(data)

    return data


def model_init():
    return AutoModelForSequenceClassification.from_pretrained("allenai/biomed_roberta_base", return_dict=True)

def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "weight_decay" : tune.loguniform(1e-3, 1e-1),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32]),
    }


def main():
    
    config = {
        "categories" : ["Machine Learning", "Deep Learning", "Attention-based", "Docking", "Classification", "Regression", "Cross-validation", "Independent testing", "SOTA comparison", "Experimental validation", "Accuracy", "RMSE", "AUROC", "Spearman/Pearson", "F1-score", "Other"]
        "data_dir" : "../data/train"
    }

    pt_model_name = "allenai/biomed_roberta_base"
    num_epochs = 10
    
    id2label = {0: "NO", 1: "YES"}
    label2id = {"NO": 0, "YES": 1}
    model = AutoModelForSequenceClassification.from_pretrained(pt_model_name, num_labels=2, id2label=id2label, label2id=label2id)

    for categ in config["categories"]:
        wandb.init(project="expert_opinion", name=categ, group=categ)

        df = pd.read_csv(f"{config['data_dir']}/train_{categ[0:4]}.csv")
        df = df.sample(frac=1, random_state=42)
        data = format_data(df, categ)
        tokenizer = RobertaTokenizer.from_pretrained(pt_model_name, padding=True, truncation=True, do_lower_case=True)
        tokenized_data = data.map(preprocess, batched=True, fn_kwargs={"tokenizer":tokenizer})
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        splits = folds.split(np.zeros(tokenized_data.num_rows), tokenized_data["label"])
        dataset_list = []
        for train_i, val_i in splits:
            temp_train = tokenized_data.select(train_i)
            temp_val = tokenized_data.select(val_i)
            dataset_list.append((temp_train, temp_val))
    
        accuracies = []
        f1s = []
        for i in range(1):
            model = AutoModelForSequenceClassification.from_pretrained(pt_model_name, num_labels=2, id2label=id2label, label2id=label2id)
            training_args = TrainingArguments(
                output_dir=f"{categ[0:4]}_{i}_tuning",
                save_total_limit=1,
                num_train_epochs=num_epochs,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                eval_steps=1,
                report_to="wandb",
                run_name=f"{categ[0:4]}",
                push_to_hub=False
            )
            trainer = Trainer(
                args=training_args,
                train_dataset=dataset_list[i][0],
                eval_dataset=dataset_list[i][1],
                tokenizer=tokenizer,
                data_collator=data_collator,
                model_init=model_init,
                compute_metrics=compute_metrics,
            )
    
            trainer.hyperparameter_search(
                direction="maximize",
                backend="ray",
                hp_space=ray_hp_space,
                n_trials=20,
                local_dir="../../huggingface_models/ray_cache")
                
        wandb.finish()
        
main()