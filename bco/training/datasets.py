import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd import grad
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
op = os.path
import json
import re


def process_data(filename, input_regex=None, 
                output_regex='^sqJ$', 
                input_columns=None,
                output_columns=None):
    try:            
        data = pd.read_csv(filename)
    except FileNotFoundError:
        data = pd.read_csv('../'+filename)

    # Identify relevant columns
    names = list(data.columns)
    icol = input_columns if input_columns else [n for n in names if bool(re.match(input_regex, n))]
    icol_star = [vl + '_star' for vl in icol]
    dcol = ["dsqJ_" + x for x in icol]

    # Add delta xstar data
    dlcol = [vl + '_delta' for vl in icol]
    data[dlcol] = pd.DataFrame(data[icol].values - data[icol_star].values)

    # Identify output columns
    ocol = output_columns if output_columns else [n for n in list(data.columns) if bool(re.match(output_regex, n))]

    # Infeasible data tensors
    D = data[data.classes < 1/2].copy() # get infeasible points
    ifs = torch.tensor(D[icol].values)
    ifs_star = torch.tensor(D[icol_star].values)
    out = torch.tensor((D[ocol].values))
    dout = torch.tensor(D[dcol].values)

    # Feasible data tensors
    D = data[data.classes > 1/2].copy() # get feasible points
    fs = torch.tensor(D[icol].values)

    return (fs, ifs, ifs_star, out, dout)


class BaseDataset():

    def __init__(self, filename, params, output_regex, device=None):
        input_col = {k:params[k] for k in ['input_regex', 'input_columns'] if k in params}
        (fs, ifs, ifs_star, out, dout) = process_data(filename,
                    output_regex=output_regex,
                    **input_col)
        self.device = device
        self.tensors = (fs, ifs, ifs_star, out, dout)

        # localize infeasible points (which are not xStar)
        self.n_infeasible = ifs.shape[0]
        self.n_feasible = fs.shape[0]
        self.n_total = self.n_feasible + self.n_infeasible

        # define input and output mean
        self.input_mean = (ifs.sum(0) + fs.sum(0)) / self.n_total
        self.output_mean = out.mean(0)
        if self.n_total > 1:
            self.input_std = ((ifs - self.input_mean).square().sum(0) + \
                                (fs - self.input_mean).square().sum(0)).div(self.n_total - 1).sqrt()
        else:
            self.input_std = torch.ones(1)
        if self.n_infeasible > 1:
            self.output_std = out.std(0)
        else:
            self.output_std = torch.ones(1)

        # batching
        # n_slices = np.maximum( np.maximum(self.n_feasible,  self.n_infeasible) // params['optim']['batch_size'], 1)
        # n_slices = min([params['optim']['n_batch'], max([self.n_feasible, self.n_infeasible])])
        n_slices = min([params['optim']['n_batch'], max([self.n_feasible, self.n_infeasible])])
        self.n_slices = n_slices
        
        self.fs_index = np.arange(fs.shape[0])
        self.ifs_index = np.arange(ifs.shape[0])
        
        fs_batch_size = max([1, self.n_feasible // (n_slices)])
        ifs_batch_size = max([1, self.n_infeasible // (n_slices)])
        
        self.fs_slices = [((n_slices-1)*fs_batch_size, fs.shape[0])]
        self.ifs_slices = [((n_slices-1)*ifs_batch_size, ifs.shape[0])]
        for j in range(n_slices-1):
            self.fs_slices.append( (j* fs_batch_size, (j+1)* fs_batch_size))
            self.ifs_slices.append((j*ifs_batch_size, (j+1)*ifs_batch_size))
        
        # anchors
        self.anchors = None
        self.anchor_engine = torch.quasirandom.SobolEngine(dimension=ifs.shape[1], scramble=True)
        if 'bounds' in params:
            self.bound = torch.tensor(params['bounds'])
        self.n_anchors = params['sdf_regularization_anchors']
        self.fixed_anchors = params['fixed_regularization_anchors']
        if 'random_anchor_walk' in params:
            self.random_anchor_walk = params['random_anchor_walk']
        else:
            self.random_anchor_walk = None
        if self.random_anchor_walk is not None:
            self.random_anchor_walk = torch.tensor(self.random_anchor_walk)

    def to(self, device):
        self.tensors = [t.to(device) for t in self.tensors]
        self.input_mean = self.input_mean.to(device)
        self.output_mean = self.output_mean.to(device)
        self.input_std = self.input_std.to(device)
        self.output_std = self.output_std.to(device)
        self.bound = self.bound.to(device)
        if self.random_anchor_walk is not None:
            self.random_anchor_walk = self.random_anchor_walk.to(device)
        self.device = device

    def copy_stats(self, source):
        self.input_mean.data = source.input_mean.data
        self.output_mean.data = source.output_mean.data
        self.input_std.data = source.input_std.data
        self.output_std.data = source.output_std.data

    def get_batches(self, shuffle=True):
        if self.n_slices == 1:
            batches = [self.tensors]
        else:
            if shuffle:
                np.random.shuffle(self.fs_index)
                np.random.shuffle(self.ifs_index)
            fs, ifs, ifs_star, out, dout = self.tensors
            batches = []

            for s in range(self.n_slices):
                sl = self.fs_slices[s]
                isl = self.ifs_slices[s]
                batches.append([fs[sl[0]:sl[1]],
                        ifs[isl[0]:isl[1]],
                        ifs_star[isl[0]:isl[1]],
                        out[isl[0]:isl[1]],
                        dout[isl[0]:isl[1]]])
        return batches

    def get_dataset(self, use_xStar=False):
        fs, ifs, ifs_star, out, dout = self.tensors

        if use_xStar:
            raise NotImplementedError
        else:
            input = torch.cat((fs, ifs))
            
            classes = torch.zeros(input.size(0))
            classes[:fs.shape[0]] = 1

            output = torch.zeros((input.size(0),1))
            output[fs.shape[0]:] = out

            doutput = torch.zeros(input.shape)
            doutput[fs.shape[0]:] = dout
        
        return input, output, doutput, classes
    
    def get_anchors(self):
        if (self.anchors is None or self.fixed_anchors is False) and self.n_anchors > 0:
            self.anchors = (self.anchor_engine.draw(self.n_anchors).double().to(self.device) * 
                            (self.bound[1:] - self.bound[:1]) + self.bound[:1])
        if hasattr(self,"random_anchor_walk") and self.random_anchor_walk is not None:
            self.anchors.add_(torch.randn_like(self.anchors)*self.random_anchor_walk)
            self.anchors.add_((self.anchors < self.bound[:1]) * (self.anchors - (self.bound[:1] - self.bound[-1:])))
            self.anchors.add_((self.anchors > self.bound[-1:]) * (self.anchors + (self.bound[:1] - self.bound[-1:])))
        return self.anchors
        
    def get_sqJ(self, output):
        raise NotImplementedError

    def get_J(self, output):
        return self.get_sqJ(output) ** 2


class sqJDataset(BaseDataset):
    def __init__(self, filename, params):
        super().__init__(filename, params, output_regex='^sqJ$')

    def get_sqJ(self, output):
        return (F.relu(output)) 


class xStarDataset(BaseDataset):
    def __init__(self, filename, params):
        super().__init__(filename, params,
                        output_regex='.+_delta$')

    def get_sqJ(self, output_tensor):
        return torch.sqrt(((output_tensor)**2).sum(1))


def build_dataset(params, test=False):
    if params['model_type'] in ['sqJ_classifier_w_derivative', 'sqJ_orth_cert', 'sqJ_hinge_classifier', 'sqJ', 'classifier']:
        dataset = sqJDataset(params['filename'], params)
        test_dataset = sqJDataset(params['test_filename'], params)
        test_dataset.copy_stats(dataset)
        return dataset, test_dataset

    elif params['model_type'] == 'xStar':
        dataset = xStarDataset(params['filename'], params)
        test_dataset = xStarDataset(params['test_filename'], params)
        test_dataset.copy_stats(dataset)
        return dataset, test_dataset
    else:
        raise NotImplementedError
