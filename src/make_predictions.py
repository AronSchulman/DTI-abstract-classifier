import numpy as np
import pandas as pd
import json
from transformers import pipeline
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def main():
    
    with open("../user_prediction_config.json", 'r') as file:
        cfg = json.load(file)

    id2label = {0: "NO", 1: "YES"}
    label2id = {"NO": 0, "YES": 1}
    
    categories = cfg["categories"]
    for cat in categories:
        df_abs = pd.read_csv(cfg["input_file"])
        pipes = {}
        for i in range(5):
            pipes[f"{cat}_{i}"] = pipeline(model=f"BaronSch/{cat[0:4]}_{i}")
            
        categ_names = list(pipes.keys())
        df_abs[categ_names] = 0
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
        num_abs = len(df_abs)
        for i in range(num_abs):
            for categ in pipes:
                out = pipes[categ](df_abs.text[i], **tokenizer_kwargs)
                df_abs.at[i, categ] = label2id[out[0]["label"]]
        df_majority = df_abs[categ_names]
        for j in [3, 4, 5]:
            df_abs[f"majority_{j}"] = (df_majority.sum(axis=1) >= j).astype(int)
        df_abs.to_csv(f"{cfg['save_path']}/{num_abs}_predicted_majorities_{cat[0:4]}.csv", index=False)
    
main()