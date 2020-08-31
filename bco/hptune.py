from ray import tune
from bco.train import train
import json
from itertools import product
import pandas as pd
import numpy as np

with open('twins_orth.json', 'r') as f:
    params = json.load(f)

def train_(config):
    with open('twins_orth.json', 'r') as f:
        params = json.load(f)
    params.update(config)
    train(params, tune_search=True)

def multio(config):
    with open('multio_orth.json', 'r') as f:
        params = json.load(f)
    params.update(config)
    train(params, tune_search=True)

analysis = tune.run(
    train_multio,
    config={
        "model":{
            "hidden_layers": tune.grid_search([[l]*n for l, n in product([16,32], [3,4,5])])
        },
        "optim":{
            "step_size": tune.sample_from(lambda spec: np.exp(np.random.uniform(-4,-1.5) * np.log(10)))
        }
    },
    num_samples=15)

print("Best config: ", analysis.get_best_config(metric="mean_loss"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
df.to_csv('hptune_orth.csv', index=False)
