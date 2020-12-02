import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
torch.set_default_dtype(torch.float64)
import os
from train import train
import json
from joblib import Parallel, delayed
from itertools import product
import re


shared_params = {'model_restart': None,
 'filename_suffix': 'nrm',
 'hidden_layers': [128,128,128],
 'step_size': 0.001,
 'epochs': 30000,
 'batch_size': 25,
 'weight_decay': 0.,
 'activation': 'groupsort8',
'filename': 'data/multio_50/multio_2d_n0.csv',
'test_filename': 'data/multio_50/test_multio_2d.csv',
 'input_regex': '^x\\d+$',
 'model_type': 'sqJ_classifier_w_derivative',
 'seed':None,
 'SWA':False}


def rerun(f):
    params = shared_params.copy()
    with open('models/'+f, 'r') as ff:
        p_ = json.load(ff)
    params.update(p_)
    train(params)


def extract_dataFrame():
    files = [f[:-5] for f in os.listdir('models') if f.endswith('.json')]
    C = []
    for f in files:
        try:
            with open('models/' + f + '.json', 'r') as ff:
                C.append(json.load(ff))
        except:
            print("Failed opening ", f)
            continue
        C[-1]['name'] = f
    C = pd.DataFrame(C)
    C.to_csv('models/summary.csv', index=False)


if __name__ == "__main__":
    PARAMS = []
    for st in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        params = shared_params.copy()
        # params['activation'] = 'groupsort4'
        # params['hidden_layers'] = hl
        params['step_size'] = st
        params['filename_suffix'] = str(st)
        PARAMS.append(params)


    Parallel(n_jobs=8)(delayed(train)(p) for p  in PARAMS)
    extract_dataFrame() 
