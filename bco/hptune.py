from ray import tune
from bco.train import train
import json
from itertools import product
import pandas as pd
import numpy as np


with open('multio.json', 'r') as f:
    params_multio = json.load(f)

with open('twins_orth.json', 'r') as f:
    params_twins = json.load(f)

def twins(config, checkpoint_dir='twins'):
    params_twins.update(config)
    train(params_twins, tune_search=True)

def multio(config, checkpoint_dir='multio'):
    params_multio.update(config)
    train(params_multio, tune_search=True)

analysis = tune.run(
   multio,
    config={
        "model":{
            "hidden_layers": tune.grid_search([[l]*n for l, n in product([2], [4,5,6,7])])
        },
        "optim":{
            "step_size": tune.sample_from(lambda spec: np.exp(np.random.uniform(-4,-1.5) * np.log(10)))
        },
        "rbf":  tune.grid_search([3,5,7,9,11])
    },
    num_samples=15,
    raise_on_failed_trial=False,
    local_dir='ray_results/'
    )

print("Best config: ", analysis.get_best_config(metric="mean_loss"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
df.to_csv('hptune_orth.csv', index=False)
