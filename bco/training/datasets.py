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


def process_data(filename, input_regex, output_regex='^sqJ$', 
                use_xStar=False, 
                xStarClass = 'infeasible'
                ):
                
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        data = pd.read_csv('../' + filename )

    # identify relevant columns
    names = list(data.columns)
    icol = [n for n in names if bool(re.match(input_regex, n))]
    icol_star = [vl + '_star' for vl in icol]
    dcol = ["dsqJ_" + x for x in icol]

    # add delta xstar data
    dlcol = [vl + '_delta' for vl in icol]
    data[dlcol] = pd.DataFrame(data[icol].values - data[icol_star].values)

    # Identify output columns
    ocol = [n for n in list(data.columns) if bool(re.match(output_regex, n))]

    # # Add x* data
    if use_xStar:
        D = data[data.classes < 1/2].copy() # get infeasible points
        D[icol] = D[icol_star].values
        D['sqJ'] = 0.
        D['J'] = 0.
        D[dlcol] = 0.
        # same classes as original point: infeasible
        if xStarClass == 'infeasible':
            D['classes'] = 0.
            # same sqrt derivatives as original point
        elif xStarClass == 'feasible':
            D['classes'] = 1.
            D[dcol] = 0.
        elif xStarClass == 'both':
            D_ = D.copy()
            D['classes'] = 1.
            D_['classes'] = 0.
            D = pd.concat((D, D_))
        else:
            raise NotImplementedError
        data = pd.concat((data, D))
        # data.sample(frac=1) #shuffle

    # Extract tensors
    inp = torch.tensor(data[icol].values)
    out = torch.tensor((data[ocol].values))
    dout = torch.tensor(data[dcol].values)
    classes = torch.tensor(data[['classes']].values)

    # Add random points in regions of interest
    # if use_xStar:
    #     N = 15
    #     # delt = torch.tensor(data[dlcol].values)
    #     # delt = delt.repeat(repeats=(N,1))
    #     X = 2*torch.rand((inp.shape[0]*N, inp.shape[1])) - 1
    #     X = X / X.norm(dim=-1, keepdim=True)
    #     new_inp = inp.repeat(repeats=(N,1))
    #     new_inp = new_inp + out.repeat(repeats=(N,1)) * X
    #     inp = torch.cat((inp, new_inp))
    #     out = out.repeat(repeats=(N+1,1))
    #     dout = dout.repeat(repeats=(N+1,1))
    #     new_classes = 2*classes.repeat(repeats=(N,1)) - 1 
    #     classes = torch.cat((classes, new_classes))

    return (inp, out, dout, classes)


class BaseDataset(TensorDataset):
    r"""Dataset wrapping tensors.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, filename, params, output_regex, use_xStar=False, xStarClass = 'infeasible', 
                input_mean=None, output_mean=None, input_std=None, output_std=None, augment=False):
        (inp, out, dout, classes) = process_data(filename,
                    input_regex=params['input_regex'],
                    output_regex=output_regex,
                    use_xStar=use_xStar, 
                    xStarClass=xStarClass)

        self.input_mean = inp.mean(0) if input_mean is None else input_mean
        self.output_mean = out.mean(0) if output_mean is None else output_mean
        self.input_std = inp.std(0) + 1e-10 if input_std is None else input_std
        self.output_std = out.std(0) + 1e-10 if output_std is None else output_std

        # localize infeasible points (which are not xStar)
        strict_infeasible = torch.logical_and(self.get_J(out) > 1e-4, classes == 0).squeeze()
        self.n_infeasible = strict_infeasible.sum()
        self.augment = augment
        if augment:
            inp = torch.cat((inp, inp[strict_infeasible]))
            out = torch.cat((out, out[strict_infeasible]))
            dout = torch.cat((dout, dout[strict_infeasible]))
            classes = torch.cat((classes, (2 * classes - 1)[strict_infeasible]))

        super().__init__(inp, out, dout, classes)
        self.index = np.arange(inp.shape[0])
        n = np.maximum(inp.shape[0] // params['optim']['batch_size'], 1)
        self.slices = [(j*params['optim']['batch_size'], (j+1)*params['optim']['batch_size']) for j in range(n)]


    def get_batches(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self.index)
        if self.augment:
            inp, out, dout, classes = self.tensors
            inp_ = inp.clone()
            X = 2 * torch.rand((self.n_infeasible, inp.shape[1])) - 1
            Xnrm = torch.rand((self.n_infeasible, 1))
            X = X / X.norm(dim=-1, keepdim=True) * Xnrm
            inp_[-self.n_infeasible:] -=  out[-self.n_infeasible:] * X
            return [(inp_[self.index[s[0]:s[1]]], out[self.index[s[0]:s[1]]], dout[self.index[s[0]:s[1]]], classes[self.index[s[0]:s[1]]])
                        for s in self.slices]
        else:
            return [(t[self.index[s[0]:s[1]]] for t in self.tensors) 
                        for s in self.slices]
        

    def get_sqJ(self, output):
        raise NotImplementedError

    def get_J(self, output):
        return self.get_sqJ(output)**2


class sqJDataset(BaseDataset):
    def __init__(self, filename, params, use_xStar=False, xStarClass = 'infeasible', 
                input_mean=None, output_mean=None, input_std=None, output_std=None, augment=False):
        super().__init__(filename, params, output_regex='^sqJ$',
                        use_xStar = use_xStar, xStarClass=xStarClass, 
                        input_mean=input_mean, 
                        output_mean=torch.zeros(1), 
                        input_std=input_std, 
                        output_std=output_std,
                        augment=augment)
            

    def get_sqJ(self, output):
        return (F.relu(output)) 


class xStarDataset(BaseDataset):
    def __init__(self, filename, params, use_xStar=False, xStarClass = 'infeasible', input_mean=None, output_mean=None, input_std=None, output_std=None):
        super().__init__(filename, params, 
                        use_xStar=use_xStar, xStarClass=xStarClass,
                        output_regex='.+_delta$',
                        input_mean=input_mean,
                        output_mean=output_mean,
                        input_std=input_std,
                        output_std=output_std)

    def get_sqJ(self, output_tensor):
        return torch.sqrt(((output_tensor)**2).sum(1))


def build_dataset(params, test=False):
    use_xStar_test = False
    if test:
        use_xStar_train = False
        augment_train = False
    else:
        use_xStar_train = True
        augment_train = params['augment'] 
    if params['model_type'] in ['sqJ_classifier_w_derivative', 'sqJ_orth_cert']:
        dataset = sqJDataset(params['filename'], params, use_xStar=use_xStar_train, 
                                xStarClass='infeasible', augment=augment_train)
        test_dataset = sqJDataset(params['test_filename'], params, 
                                        use_xStar=use_xStar_test,
                                        xStarClass='infeasible',
                                        input_mean = dataset.input_mean, 
                                        output_mean = dataset.output_mean, 
                                        input_std = dataset.input_std, 
                                        output_std = dataset.output_std,
                                        augment=False)
        return dataset, test_dataset
    elif params['model_type'] in ['sqJ']:
        dataset = sqJDataset(params['filename'], params, use_xStar=use_xStar_train, 
                                xStarClass='feasible', augment=augment_train)
        test_dataset = sqJDataset(params['test_filename'], params, 
                                        use_xStar=use_xStar_test,
                                        xStarClass='feasible',
                                        input_mean = dataset.input_mean, 
                                        output_mean = dataset.output_mean, 
                                        input_std = dataset.input_std, 
                                        output_std = dataset.output_std,
                                        augment=False)
        return dataset, test_dataset
    elif params['model_type'] == 'classifier':
        dataset = sqJDataset(params['filename'], params, 
                                use_xStar=use_xStar_train, xStarClass='both', augment=augment_train)
        test_dataset = sqJDataset(params['test_filename'], params, use_xStar=use_xStar_test,
                                        input_mean = dataset.input_mean, 
                                        output_mean = dataset.output_mean, 
                                        input_std = dataset.input_std, 
                                        output_std = dataset.output_std,
                                        augment=False)
        return dataset, test_dataset
    elif params['model_type'] == 'xStar':
        dataset = xStarDataset(params['filename'], params, use_xStar=use_xStar_train, augment=augment_train)
        test_dataset = xStarDataset(params['test_filename'], params, use_xStar=use_xStar_test,
                                        input_mean = dataset.input_mean, 
                                        output_mean = dataset.output_mean, 
                                        input_std = dataset.input_std, 
                                        output_std = dataset.output_std,
                                        augment=False)
        return dataset, test_dataset
    else:
        raise NotImplementedError