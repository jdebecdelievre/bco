import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import grad
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
op = os.path
import json
import pdb
from tqdm import tqdm
from collections import defaultdict

from ray import tune

from bco.training.models import build_model, scalarize
from bco.training.datasets import build_dataset
from bco.training.losses import loss_calc
from flatten_dict import flatten, unflatten
from bco.training import default_params
from bco.training.losses import parse_coefs
from bco.training.optimizers import get_optimizer

# Experimental
from bco.training.early_stopping import EarlyStopping

import argparse

def process_params(params):
    # Fill params
    params = update_default_dict(params)

    # Restart model
    if params['model_restart']:
        model_file = params['model_restart']
        with open(op.join("models", params['model_restart']+'.json'), "r") as f:
            params = json.load(f)
            params['model_restart'] = model_file

    # Seed
    if params['seed'] is None:
            params['seed'] = np.random.choice(2147483648) # 2^31
    
    # Hidden layers
    if type(params['model']['hidden_layers']) == str:
        params['model']['hidden_layers'] = [int(h) for h in params['model']['hidden_layers'].strip(']').strip('[').split(',')]
    
    # Set random seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # Create folder for models
    os.makedirs('models', exist_ok=True)

    return params

def train(params={}, tune_search=False):
    # Process params
    params = process_params(params)

    # Load data
    dataset, test_dataset = build_dataset(params)
    params['model']['input_size'] = dataset.input_mean.shape[-1]
    params['input_size'] = dataset.input_mean.shape[-1]
    params['train_set_size'] = dataset.tensors[0].shape[0]

    # Build model
    model = build_model(params)
    if params['normalize_input']:
        model.input_mean.data = dataset.input_mean.data
        model.input_std.data = dataset.input_std.data
    if params['normalize_output']:
        model.output_mean.data = dataset.output_mean.data
        model.output_std.data = dataset.output_std.data

    # Choose opt
    opt, scheduler = get_optimizer(params['optim'], model)

    # Add stochastic weight averaging
    if params['SWA']:
        from torchcontrib.optim import SWA
        opt = SWA(opt, swa_start=8000, swa_freq=5, swa_lr=1e-3)

    # Load pretrained model
    if params['model_restart']:
        model.load_state_dict(torch.load(op.join("models", params['model_restart'] + ".mdl")))
        opt.load_state_dict(torch.load(op.join("models", params['model_restart'] + ".opt")))
    
    # Prepare tensorboard logging
    test_writer = SummaryWriter(comment=("_" + params['filename_suffix'] + '_test') if params['filename_suffix'] else "_test")
    train_writer = SummaryWriter(comment=("_" + params['filename_suffix'] + '_train') if params['filename_suffix'] else "_train")
    if tune_search:
        test_writer.close()
        train_writer.close()
    basename = os.path.basename(test_writer.log_dir[:-5])
    print('Training', basename)

    # Set up early stopping
    if params['early_stopping'] is not None:
        stop = EarlyStopping(patience=params['early_stopping'],  mode='min')

    # Device
    device = 'cpu'
    coefs = parse_coefs(params, device)

    # Train
    model.train()
    f1_score = -5
    print(json.dumps(params, indent=4, sort_keys=True))
    for e in tqdm(range(params["epochs"])):
        L = defaultdict(float)

        B = dataset.get_batches(shuffle=True)
        for i, o, do, cl in B:
            opt.zero_grad()
            loss, loss_dict = loss_calc(i, o, do, cl, model, params, coefs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['optim']['max_grad_norm'])
            opt.step(lambda: float(loss))

            with torch.no_grad():
                for l in loss_dict:
                    L[l] += loss_dict[l].data
                L['loss'] += loss.data

                model.logg_gradient_norm(train_writer)
                model.projection('update')

        model.logg_gradient_norm(train_writer, epoch=e)
        model.projection('epoch')

        if e%50 ==0:
            # Log train metrics
            model.eval()
            with torch.no_grad():
                # params.update(model.metrics(i, o, cl, train_writer, e, 'train_'))
                train_writer.add_scalar('loss/loss', L['loss'], e)
                for l in L:
                    train_writer.add_scalar('loss/'+l, L[l], e)
            model.train()

        if e%100 == 0:
            # Write loss to params
            logs = {}
            logs['loss'] = scalarize(L['loss'])
            for l in L:
                logs['loss_'+l] = scalarize(L[l])

            # Save test metrics
            model.eval()
            with torch.no_grad():
                test_input, test_output, _, test_classes = test_dataset.tensors
                logs.update(model.metrics(test_input, test_output, test_classes, test_writer, e, 'test_'))
            if tune_search:
                tune.track.log(**logs)
            params.update(logs)
            # Log matrix eigenvalues
            # if not model.params['per_update_proj']['turned_on'] and not model.params['per_epoch_proj']['turned_on']:
            #     model.log_sing(train_writer)

            # Save model
            model_file_name = os.path.join("models", basename + ".mdl")
            opt_file_name = model_file_name[:-4]+'.opt'
            torch.save(model.state_dict(), model_file_name)
            torch.save(opt.state_dict(), opt_file_name)
            if not tune_search:
                with open(model_file_name[:-4]+'.json', "w") as f:
                    json.dump(params, f, indent=4)

            # LR scheduler
            if scheduler is not None:
                scheduler.step(loss)
                LR = torch.tensor([group['lr'] for group in opt.param_groups]).mean()
                train_writer.add_scalar('learning_rate', LR, e)
                if LR< 1e-7:
                    print(f'Stopping Criterion reached at epoch {e}: lr = {LR}')
                    break

            # Stopping criterion
            f1_score = params['test_f1'] if not np.isnan(params['test_f1']) else 0
            if params['early_stopping'] is not None and (stop.step(torch.tensor(params['test_Jrmse']))):
                break
                
            model.train()
            
            
    if params['SWA']:
        opt.swap_swa_sgd()

    print(params)
    return params, model_file_name


def update_default_dict(params):
    from bco.training import default_params

    # Flatten
    default_params = flatten(default_params)
    params = flatten(params)

    # Check entries
    for p in params:
        assert p in default_params, f"Invalid parameter {p}"
    
    # Update
    default_params.update(params)

    return unflatten(default_params)


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params_file", type=str, default='params.json')
parser.add_argument("-fs", "--filename_suffix", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args() 
    with open(args.params_file, 'r') as f:
        params = json.load(f)
    if args.filename_suffix:
        params['filename_suffix'] = args.filename_suffix

    train(params)