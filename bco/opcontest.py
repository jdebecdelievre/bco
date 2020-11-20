import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
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
from bco.training.datasets import sqJDataset
from bco.training.losses import loss_calc
from flatten_dict import flatten, unflatten
from bco.training import default_params
from bco.training.losses import parse_coefs
from bco.training.optimizers import get_optimizer


import argparse

import torch
from scipy import optimize
import torch.nn.functional as F
import math
import numpy as np
from functools import reduce
from collections import OrderedDict

from scipy import optimize
from pyoptsparse import Optimization, OPT, History
from copy import deepcopy

device = 'cpu'

import ipdb;
from train import train

class Objective(nn.Module):
    def __init__(self, params):
        super(Objective, self).__init__()
        self.params = params

        # Load data
        self.dataset = sqJDataset(params['filename'], params, use_xStar=True, 
                         xStarClass='feasible', augment=params['augment'])
        self.params['model']['input_size'] = self.dataset.input_mean.shape[-1]
        self.params['input_size'] = self.dataset.input_mean.shape[-1]
        self.params['train_set_size'] = self.dataset.tensors[0].shape[0]

        # Build model
        model = build_model(self.params)
        if params['normalize_input']:
            model.input_mean.data = self.dataset.input_mean.data
            model.input_std.data = self.dataset.input_std.data
        if params['normalize_output']:
            model.output_mean.data = self.dataset.output_mean.data
            model.output_std.data = self.dataset.output_std.data
        self.model = model
        self.coefs = parse_coefs(params, device)
        
        # only one batch dataset
        B = self.dataset.get_batches(shuffle=False)
        self.B = [d_ for d_ in B[0]]

    def forward(self):
        i, o, do, cl = self.B
        loss, loss_dict = loss_calc(i, o, do, cl, self.model, self.params, self.coefs)
        return loss
    
    def constraint(self):
        i, o, do, cl = self.B
        clsign = (2 * cl - 1)
        # ipdb.set_trace()
        # def fun(*args):
        #     # self.model.load_state_dict(w)
        #     for p, pm in zip(self.model.parameters(), args):
        #         p.data = pm
        #     return self.model._net(i)
        # J = jacobian(fun, tuple(self.model.parameters()))
        self.zero_grad()

        Jdata = {k: np.zeros((i.shape[0], p.flatten().shape[0])) 
                for k, p in self.named_parameters() if p.requires_grad}
        
        ndim = i.shape[1]
        Jderiv = [{k:np.zeros_like(j) for k,j in Jdata.items()} for d in range(ndim)]
                
        o_ = torch.zeros(o.shape)
        do_ = torch.zeros(do.shape)
        deriv = torch.zeros(do.shape)
        for ind in range(i.shape[0]):

            ici = i[ind:ind+1]
            ici.requires_grad = True

            oci = self.model(ici)
            o_[ind] = oci

            # Direct prediction
            oci.backward()
            # oci.backward(create_graph=True)
            for k, p in self.named_parameters():
                if p.requires_grad:
                    Jdata[k][ind] = (clsign[ind] * p.grad).detach().flatten().numpy()
                    p.grad *=0
            
            # Derivative prediction
            # for d in range(ndim):
            #     ici = i[ind:ind+1]
            #     ici.requires_grad = True
            #     self.model.zero_grad()

            #     oci = self.model(ici)
            #     doci = grad(oci, [ici], create_graph=True)[0].squeeze()

            #     # do_[ind] = ici.grad.detach()
            #     do_[ind] = doci.detach()
            #     # doci = ici.grad.squeeze()
            #     # self.model.zero_grad()
            #     # ipdb.set_trace()
            #     drv = (do[ind, d] ** 2 - doci[d] **  2)
            #     deriv[ind, d] = drv
            #     drv.backward()
            #     # drv.backward(retain_graph=True)
            #     for k, p in self.named_parameters():
            #         if p.requires_grad:
            #             if p.grad is not None:
            #                 Jderiv[d][k][ind] = p.grad.flatten().numpy()
            #                 p.grad *= 0
            
        funcs =  {
            'data': ((o_ - o) * clsign).detach().squeeze().numpy(),
            'obj':0.,
            'grads': {'data': Jdata, 'obj': {k:Jdata[k][0]*0 for k in Jdata}}
        }

        # for d in range(ndim):
        #     funcs['deriv_'+str(d)] = deriv[:, d].detach().squeeze().numpy()
        #     funcs['grads']['deriv_'+str(d)] = Jderiv[d]
        funcs['classification_loss'] = np.maximum(funcs['data'], 0).sum()

        F = 0
        self.model.zero_grad()
        for k, p in self.named_parameters():
            if p.requires_grad and 'weight' in k:
                # ipdb.set_trace()
                # funcs['grads'][k] = {k:grad(f, [p])[0].detach().flatten().numpy()}
                F = F + (p @ p.T @ p - p).square().sum()
        F.backward()
        funcs['obj'] = F.detach().numpy()
        funcs['grads']['obj'] = {k: p.grad.detach().flatten().numpy() for k, p in self.named_parameters() if p.requires_grad and 'weight' in k}
        return funcs 

params = {
    "model_restart": None,
    "filename_suffix": "twins",
    
    "model_type": "sqJ_classifier_w_derivative",
  
    "model":{
        "hidden_layers": [4,4],
        "activation": "groupsort1",
        "linear": {
            "type": "linear",
            "safe_scaling": True,
            "bjorck_beta": 0.5,
            "bjorck_iter": 15,
            "bjorck_order": 1,
            "bias": True
        },
        "per_epoch_proj": {
          "turned_on": False,
          "bjorck_beta": 0.5,
          "bjorck_iter": 20,
          "bjorck_order": 1,
          "safe_scaling": True,
          "reset_optimizer": False
        },
    
        "per_update_proj": {
          "turned_on": False,
          "bjorck_beta": 0.5,
          "bjorck_iter": 12,
          "bjorck_order": 1,
          "safe_scaling": True
        },
        "scalarize":"linear",
        },
        "optim":{
            "batch_size":150
        },
        "grad_norm_regularizer": 1.0, 
        "seed": None,
        "filename": "data/twins/twins_10_n1.csv",
        "test_filename": "data/twins/test_twins.csv",
        "input_regex": "^x\\d+$",
        "augment": False,
        "normalize_input":False,
        "normalize_output":False
  }


def train_constrained(params={}, tune_search=False):

    # torch.manual_seed(7)
    objective = Objective(params)
    objective.train()
    shapes = {k:p.shape for k, p in objective.named_parameters()}

    def objfun(x):
        x_ = {k:torch.tensor(x[k].reshape(shapes[k])) for k in x}
        objective.load_state_dict(x_, strict=False)
        objective.zero_grad()
        funcs = objective.constraint()
        return funcs, False
    
    def sensfun(x, funcs):
        return funcs['grads']

    problem = Optimization('nn', objfun)

    for k, p in objective.named_parameters():
        if p.requires_grad:
            nVars = 1
            for n in p.shape:
                nVars *= n
            problem.addVarGroup(k, nVars=nVars, value=p.detach().numpy().flatten(),
                                lower=-1., upper=1.)

    problem.addConGroup('data', upper=0, nCon=objective.B[0].shape[0])
    # for d in range(2):
    #     problem.addConGroup('deriv_'+str(d), upper=0, nCon=objective.B[0].shape[0])
    # problem.addConGroup('weights', upper=1e-6, lower=1e-6)
    # for k, p in objective.named_parameters():
    #     if p.requires_grad and 'weight' in k:
    #         problem.addCon(k, upper=0., wrt=[k])

    problem.addObj('obj')
    funcs = objective.constraint()
    ipdb.set_trace()
    name = 'train'
    opt = OPT('SNOPT', options={'Print file':f'{name}_snopt_print.out', 
                                'Summary file':f'{name}_snopt_summary.opt',
                                'Major iterations limit':300})
    # opt = OPT('IPOPT', options={
    #     "derivative_test_perturbation": 1e-10,
    #     "derivative_test": "first-order",
    #     "max_iter":3})
    sol = opt(problem, sens=sensfun, storeHistory='train.hst')

    model = objective.model
    torch.save(model.state_dict(), os.path.join("models","model.mdl"))
    with open(os.path.join("models","model.json"), "w") as f:
        json.dump(params, f, indent=4)
    torch.save(model, os.path.join("models","model.load"))
    

    print(sol.optInform)
    print(objective())
    ipdb.set_trace()         
    return params, model


if __name__ == "__main__":
    # train_scipy(params)
    train_constrained(params)
    # train_pyoptsparse(params)
    # train(params)