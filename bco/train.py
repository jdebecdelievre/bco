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
import random

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

import socket
from datetime import datetime

import torch.autograd.profiler as profiler

def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

class DummyWriter():
    def __init__(self, log_dir=None, comment=None):
        super().__init__()
        self.log_dir = log_dir
    
    def add_scalar(*args, **kwargs):
        pass


def process_params(params, dest_dir):
    # Fill params
    params = update_default_dict(params)

    # Restart model
    if params['model_restart']:
        model_file = params['model_restart']
        with open(op.join(dest_dir, "models", params['model_restart']+'.json'), "r") as f:
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
    torch.set_deterministic(True)
    np.random.seed(params['seed'])

    # Data normalization is not possible with Bjorck layers because they guarnatee a 
    # Lipshitz constant of exactly one 
    if params['normalize_input'] or params['normalize_output']:
        assert params['model']['linear'] != 'bjorck', "Bjorck layers incompatible with data normalization"
    return params

def train(params={}, tune_search=False, dest_dir='.'):
    # Process params
    params = process_params(params, dest_dir)

    # Create folder for models
    os.makedirs(op.join(dest_dir, 'models'), exist_ok=True)

    # Load data
    dataset, test_dataset = build_dataset(params)
    params['model']['input_size'] = dataset.tensors[0].shape[-1]
    params['train_set_size'] = dataset.n_feasible + dataset.n_infeasible

    # Build model
    model = build_model(params)
    if params['normalize_input']:
        model.input_mean.data = dataset.input_mean.data
        model.input_std.data = dataset.input_std.data
    if params['normalize_output']:
        model.output_mean.data = dataset.output_mean.data*0.
        model.output_std.data = dataset.output_std.data

    # Load pretrained model
    if params['model_restart']:
        model.load_state_dict(torch.load(op.join(dest_dir, "models", params['model_restart'] + ".mdl")))
        opt.load_state_dict(torch.load(op.join(dest_dir, "models", params['model_restart'] + ".opt")))
    
    # Initialize model
    # try:

    if params['bias_init'] == True:
        datapoints, _, _, _ = dataset.get_dataset()
        # idx = random.sample(range(datapoints.size(0)), len(params['model']['hidden_layers']) + 1)
        idx = torch.randint(datapoints.size(0), (len(params['model']['hidden_layers']) + 1,))
        model._net.reset_parameters(datapoints=datapoints[idx])

    # except KeyError:
    #     pass
    # Choose opt
    opt, scheduler = get_optimizer(params['optim'], model)

    # Add stochastic weight averaging
    # if params['SWA']:
    #     from torchcontrib.optim import SWA
    #     opt = SWA(opt, swa_start=8000, swa_freq=5, swa_lr=1e-3)
    
    # Prepare tensorboard logging
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(dest_dir, 'runs', current_time + '_' + socket.gethostname())
    if tune_search:
        test_writer = DummyWriter( log_dir= log_dir + \
                        ("_" + params['filename_suffix'] + '_test') if params['filename_suffix'] else "_test")
        train_writer = DummyWriter(log_dir=log_dir + \
                        ("_" + params['filename_suffix'] + '_train') if params['filename_suffix'] else "_train")
    else:
        test_writer = SummaryWriter(log_dir=log_dir + \
                        ("_" + params['filename_suffix'] + '_test') if params['filename_suffix'] else "_test")
        train_writer = SummaryWriter(log_dir=log_dir + \
                        ("_" + params['filename_suffix'] + '_train') if params['filename_suffix'] else "_train")
    basename = os.path.basename(test_writer.log_dir[:-5])

    # Set up early stopping
    if params['early_stopping'] is not None:
        stop = EarlyStopping(patience=params['early_stopping'],  mode='min')

    # Device
    device = 'cpu'
    coefs = parse_coefs(params, device)

    # Train
    model.train()
    f1_score = -5
    stop_here = False
    if tune_search:
        iterator = range(params["epochs"])
    else:
        print('Training', basename)
        print(json.dumps(params, indent=4, sort_keys=True))
        print(model)
        iterator = tqdm(range(params["epochs"]))
    for e in iterator:
        L = defaultdict(float)

        B = dataset.get_batches(shuffle=dataset.n_slices > 1)
        for batch in B:
            anchors = dataset.get_anchors()
            opt.zero_grad(set_to_none=True)
            loss, loss_dict = loss_calc(batch, anchors, model, params, coefs)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), params['optim']['max_grad_norm'])
            if params['optim']['optimizer'] == 'adahessian':
                loss.backward(create_graph=True)
                _, gradsH = get_params_grad(model)
                opt.step(gradsH)
            else:
                loss.backward()
                opt.step(lambda: float(loss))
            # import ipdb; ipdb.set_trace()
            with torch.no_grad():
                for l in loss_dict:
                    L[l] += loss_dict[l].data
                L['loss'] += loss.data
                model.projection('update')
                
        model.projection('epoch')

        # LR scheduler
        if scheduler is not None:
            scheduler.step(L['loss'])
            LR = torch.tensor([group['lr'] for group in opt.param_groups]).mean()
            train_writer.add_scalar('learning_rate', LR, e)
            if LR < 1e-7:
                print(f'Stopping Criterion reached at epoch {e}: lr = {LR}')
                stop_here = True

        # Stopping criterion
        f1_score = params['test_f1'] if ('test_f1' in params and not np.isnan(params['test_f1'])) else 0
        if params['early_stopping'] is not None and (stop.step(torch.tensor(params['test_Jrmse']))):
            stop_here = True

        if e%params['logging']['train'] ==0 or stop_here:
            # Write loss to params
            logs = {}
            logs['loss'] = scalarize(L['loss'])
            for l in L:
                logs['loss_'+l] = scalarize(L[l])
            
            # Save train metrics
            model.eval()
            with torch.no_grad():
                input, output, _, classes = dataset.get_dataset()
                logs.update(model.metrics(input, output, classes, train_writer, e, 'train_'))
                if tune_search:
                    tune.report(**logs)
                
                train_writer.add_scalar('loss/loss', L['loss'], e)
                for l in L:
                    train_writer.add_scalar('loss/'+l, L[l], e)
                params.update(logs)
            model.train()

        if e%params['logging']['test'] == 0 or stop_here:
            # Write loss to params
            logs = {}
            logs['loss'] = scalarize(L['loss'])
            for l in L:
                logs['loss_'+l] = scalarize(L[l])

            # Save test metrics
            model.eval()
            with torch.no_grad():
                test_input, test_output, _, test_classes = test_dataset.get_dataset()
                logs.update(model.metrics(test_input, test_output, test_classes, test_writer, e, 'test_'))
            if tune_search:
                tune.report(**logs)
            params.update(logs)

            # Save model
            model_file_name = os.path.join(dest_dir, "models", basename + ".mdl")
            opt_file_name = model_file_name[:-4]+'.opt'
            torch.save(model.state_dict(), model_file_name)
            torch.save(opt.state_dict(), opt_file_name)
            with open(model_file_name[:-4]+'.json', "w") as f:
                json.dump(params, f, indent=4)
                

            model.train()

        if stop_here:
            break            
            
    # if params['SWA']:
    #     opt.swap_swa_sgd()

    if not tune_search:
        print(params)

    return params, model_file_name, model


def update_default_dict(params):
    from bco.training import default_params

    # Flatten
    default_params = flatten(default_params)
    params = flatten(params)

    # Check entries
    for p in params:
        # assert p in default_params, f"Invalid parameter {p}"
        if p not in default_params:
            print( f"Invalid parameter {p}")
    
    # Update
    # default_params.update(params)

    # return unflatten(default_params)
    return unflatten(params)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params_file", type=str, default='params.json')
parser.add_argument("-fs", "--filename_suffix", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args() 
    with open(args.params_file, 'r') as f:
        params = json.load(f)
    if args.filename_suffix:
        params['filename_suffix'] = args.filename_suffix

    # with profiler.profile() as prof:
    #     with profiler.record_function("model_inference"):
    train(params)

        # prof.export_chrome_trace("trace.json")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))