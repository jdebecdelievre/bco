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

# Experimental
from bco.training.early_stopping import EarlyStopping

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

device = 'cpu'

import ipdb;
from train import train

class PyTorchObjective(object):
    """PyTorch objective function, wrapped to be called by scipy.optimize."""
    def __init__(self, obj_module):
        self.f = obj_module # some pytorch module, that produces a scalar loss
        # make an x0 from the parameters in this module
        parameters = OrderedDict({k:p for k, p in \
                    obj_module.named_parameters() if p.requires_grad})
        self.param_shapes = {n:parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel() 
                                   for n in parameters])

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for name, p in self.f.named_parameters():
            if p.requires_grad:
                # if p.grad is None:
                #     grad = np.zeros(p.shape)
                #     print(name, 'no grad')
                # else:
                grad = p.grad.data.numpy()
                # ipdb.set_trace()
                grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        self.f.load_state_dict(state_dict, strict=False)
        # store the raw array as well
        self.cached_x = x
        # zero the gradient
        self.f.zero_grad()
        # use it to calculate the objective
        obj = self.f()
        # backprop the objective
        obj.backward()
        self.cached_f = obj.item()
        self.cached_jac = self.pack_grads()

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_jac


class Objective(nn.Module):
    def __init__(self, params):
        super(Objective, self).__init__()
        self.params = params

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
        
        # Only one batch dataset
        B = self.dataset.get_batches(shuffle=False)
        self.B = [d_ for d_ in B[0]]

    def forward(self):
        i, o, do, cl = self.B
        loss, loss_dict = loss_calc(i, o, do, cl, self.model, self.params, self.coefs)
        return loss

params = {
    "model_restart": None,
    "filename_suffix": "twins",
    
    "model_type": "sqJ_classifier_w_derivative",
  
    "model":{
        "scalarize":"linear",
        "hidden_layers": [32,32],
        "activation": "tanh",
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
          "bjorck_iter": 5,
          "bjorck_order": 2,
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

        },
        "grad_norm_regularizer": 1.0, 
        "seed": None,
        # "filename": "data/multio_50/multio_2d_n0.csv",
        # "test_filename": "data/multio_50/multio_2d_n0.csv",
        "filename": "data/twins/twins_50_n1.csv",
        "test_filename": "data/twins/test_twins.csv",
        "input_regex": "^x\\d+$",
        "augment": False,
        "normalize_input":False,
        "normalize_output":False,

        "optim": {
        "optimizer": "adam",
        "lr_scheduler": {
            "name": "plateau",
            "patience":500,
            "factor":0.3
        },
      "momentum": 0.9,
      "weight_decay": 0.0,
      "max_grad_norm": 10000,
      "step_size": 0.01,
      "batch_size": 300
    },
    "early_stopping":None,
    "epochs":15000,
    "logging":{
      "test":100,
      "train":50
      },
    "bounds": [[-2,-2], [2,2]],
    "sdf_regularization_anchors":0,
    "boundary_anchors": 0,

    "grad_norm_regularizer": 1.0, 
    "sdf_regularizer": 1.0,
    "boundary_regularizer": 1.0
  }

def train_pyoptsparse(params={}, tune_search=False):
    # Process params
    # params = process_params(params)
    torch.manual_seed(7)
    objective = Objective(params)
    objective.train()
    shapes = {k:p.shape for k, p in objective.named_parameters()}

    # def objfun(x):
    #     x_ = {k:torch.from_numpy(x[k].reshape(shapes[k])) for k in x}
    #     objective.load_state_dict(x_, strict=False)
    # objective.zero_grad()
    #     loss = objective()
    #     loss.backward()
    #     funcs = {
    #         'obj':loss.detach().numpy(),
    #         'grads':{
    #             'obj': {k:p.grad.flatten().numpy() for k, p in objective.named_parameters() if p.requires_grad}
    #         }
    #     }
    #     # import ipdb; ipdb.set_trace()
    #     return funcs, False
    
    # def sensfun(x, funcs):
    #     return funcs['grads']
    obj = PyTorchObjective(objective)
    def objfun(x):
        # ipdb.set_trace()
        return {'obj': obj.fun(x['x'])}, 0
    
    def sensfun(x, funcs):
        return {'obj':{'x':obj.jac(x['x'])}}

    problem = Optimization('nn', objfun)
    # import ipdb; ipdb.set_trace()
    problem.addVarGroup('x', value=obj.x0, nVars=obj.x0.shape[0])
    # for k, p in objective.named_parameters():
    #     if p.requires_grad:
    #         nVars = 1
    #         for n in p.shape:
    #             nVars *= n
    #         problem.addVarGroup(k, nVars=nVars, value=p.detach().numpy().flatten())
    
    problem.addObj('obj')

    name = 'train'
    opt = OPT('SNOPT', options={'Print file':f'{name}_snopt_print.out', 
                                'Summary file':f'{name}_snopt_summary.opt',
                                'Major iterations limit':300})
    sol = opt(problem, sens=sensfun, storeHistory='train.hst')

    model = objective.model
    torch.save(model.state_dict(), os.path.join("models","model.mdl"))
    with open(os.path.join("models","model.json"), "w") as f:
        json.dump(params, f, indent=4)
    torch.save(model, os.path.join("models","model.load"))
    

    print(sol.optInform)
    print(sol.objFun(sol.xStar)[0]['obj'])
    ipdb.set_trace()         
    return params, model

def train_scipy(params={}, tune_search=False):
    # Process params
    # params = process_params(params)

    torch.manual_seed(7)

    objective = Objective(params)
    objective.train()


    maxiter = 5000
    with tqdm(total=maxiter) as pbar:
        def verbose(xk):
            pbar.update(1)
        # try to optimize that function with scipy
        obj = PyTorchObjective(objective)
        xL = optimize.minimize(obj.fun, obj.x0, method='L-BFGS-B', jac=obj.jac,
                callback=verbose, options={'ftol': 1e-8, 'gtol': 1e-8, 'disp': True,
                    'maxiter':maxiter, 'return_all':True})  

    model = objective.model
    torch.save(model.state_dict(), os.path.join("models","model.mdl"))
    with open(os.path.join("models","model.json"), "w") as f:
        json.dump(params, f, indent=4)
    torch.save(model, os.path.join("models","model.load"))
    
    # print(xL)

    import ipdb;
    ipdb.set_trace()         
    return params, model

if __name__ == "__main__":
    train_scipy(params)
    # train_constrained(params)
    # train_pyoptsparse(params)
    # torch.manual_seed(7)
    # train(params)