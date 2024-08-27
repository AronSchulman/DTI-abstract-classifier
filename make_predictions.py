import os
os.environ['HF_HOME'] = 'cache/huggingface_models/'

import pandas as pd
import yaml
from transformers import pipeline
from torch import cuda
from tqdm import tqdm
device = 'cuda' if cuda.is_available() else 'cpu'

def main():
    
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    id2label = {0: "NO", 1: "YES"}
    label2id = {"NO": 0, "YES": 1}

    df_abs = pd.read_csv(cfg["data_to_classify"])
    pipes = {}
    for i in range(cfg["cv_num_splits"]):
        pipes[f"{cfg['model_save_name']}_{i}"] = pipeline(model=f"{cfg['HF_user_name']}/{cfg['model_save_name']}_{i}")
        
    categ_names = list(pipes.keys())
    df_abs[categ_names] = 0
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    num_abs = len(df_abs)
    for i in tqdm(range(num_abs)):
        for categ in pipes:
            out = pipes[categ](df_abs.text[i], **tokenizer_kwargs)
            df_abs.at[i, categ] = label2id[out[0]["label"]]
    df_majority = df_abs[categ_names]
    for j in [3, 5]:
        df_abs[f"majority_{j}"] = (df_majority.sum(axis=1) >= j).astype(int)
    df_abs.to_csv(cfg['classification_save_dir'], index=False)

    print("Prediction complete.")
main()
