import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import grad
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
op = os.path
import json
import time


def parse_coefs(params, device='cpu'):
    if params['model_type'] == 'sqJ_classifier_w_derivative':
        return {'gradient norm': torch.tensor(params['grad_norm_regularizer'], device=device)}
    else:
        return {}

def combine_losses(loss_dict, coefs):
    loss = 0
    for l in loss_dict:
        if l in coefs:
            loss = loss + loss_dict[l] * coefs[l]
        else:
            loss = loss + loss_dict[l] 
    return loss

def loss_calc(i, o, do, cl, model, params, coefs={}):
    """Computes loss depending on modeltype

    Parameters
    ----------
    i : torch.tensor
        input tensor
    o : torch.tensor
        output tensor (target)
    do : torch.tensor
        output derivative tensor
    cl : torch.tensor
        classes
    model : torch.nn.Module
    params : dict
        dictionary of parameters
    coefs : dict, optional
        dictionary of coefficients used in the loss, such as regularization parameters, by default {}

    Returns
    -------
    torch.tensor
        loss
    """
    _i = model.normalize(input = i)
    _o = model.normalize(output=o)
    _do = model.normalize(deriv=do)
    if params['model_type'] == 'sqJ_classifier_w_derivative':
        _i.requires_grad = True
        if _i.grad is not None:
            _i.grad.detach_()
            _i.grad.zero_()

        _o_ = model._net(_i)
        _do_ = grad(_o_.sum(), [_i], create_graph=True)[0]
        _i.requires_grad = False
        # loss = (((_o - _o_).abs() + (_do - _do_).abs().sum(1, keepdim=True)).T @ (1 - cl) +  F.relu(_o_).T @ cl)
        # loss = ((_o - _o_).abs() + (_do - _do_).abs().sum(1, keepdim=True)).T @ (1 - torch.abs(cl)) +  \
        #         F.relu(_o_ * cl).T @ torch.abs(cl)
        jpred = (_o - _o_).abs().T @ (1 - torch.abs(cl))
        djpred = (_do - _do_).abs().sum(1, keepdim=True).T @ (1 - torch.abs(cl))
        feasible_class = F.relu(_o_ * cl).T @ torch.abs(cl)
        grad_norm = torch.abs(torch.norm(_do_ / (model.input_std/model.output_std), dim=1) - 1.).T.sum()

        loss_dict = {'J loss':jpred, 'dJ loss': djpred, 'classification loss': feasible_class, 'gradient norm': grad_norm}
        loss = combine_losses(loss_dict, coefs)
        return loss, loss_dict

    elif params['model_type'] == 'sqJ_orth_cert':
        _i.requires_grad = True
        if _i.grad is not None:
            _i.grad.detach_()
            _i.grad.zero_()

        _o_, certif = model._net.value_with_uncertainty(_i)
        _do_ = grad(_o_.sum(), [_i], create_graph=True)[0]
        _i.requires_grad = False

        jpred = (_o - _o_).abs().T @ (1 - torch.abs(cl))
        djpred = (_do - _do_).abs().sum(1, keepdim=True).T @ (1 - torch.abs(cl))
        feasible_class = F.relu(_o_ * cl).T @ torch.abs(cl)
        grad_norm = torch.abs(torch.norm(_do_ / (model.input_std/model.output_std), dim=1) - 1.).T.sum()
        certif_norm = torch.norm(certif, dim=1).sum()

        loss_dict = {'J loss':jpred, 
                    'dJ loss': djpred, 
                    'classification loss': feasible_class, 
                    'gradient norm': grad_norm,
                    'orthonormal certificate': certif_norm}
        loss = combine_losses(loss_dict, coefs)
        return loss, loss_dict


    elif params['model_type'] == 'sqJ_classifier':
        _o_ = model._net(_i)
        loss_dict = {'loss' : (_o - _o_).abs().T @ (1 - torch.abs(cl)) +  \
                F.relu(_o_ * cl).T @ torch.abs(cl)}
        return combine_losses(loss_dict, coefs), loss_dict
    
    elif params['model_type'] == 'sqJ':
        _i.requires_grad = True
        if _i.grad is not None:
            _i.grad.detach_()
            _i.grad.zero_()

        _o_ = model._net(_i)
        _do_ = grad(_o_.sum(), [_i], create_graph=True)[0]
        _i.requires_grad = False
        loss_dict = {'loss': ((_o - _o_).abs() + (_do - _do_).abs().sum(1, keepdim=True)).sum()}
        return combine_losses(loss_dict, coefs), loss_dict

    elif params['model_type'] == 'classifier':
        _o_ = model._net(_i)
        # loss = F.relu(1+o_.T) @ cl + F.relu(1-o_.T) @ (1. - cl)
        loss_dict = {'loss': F.softplus(2 * _o_ * cl).T @ torch.abs(cl) + F.softplus(-2 * _o_).T @ (1-torch.abs(cl))}
        return combine_losses(loss_dict, coefs), loss_dict

    elif params['model_type'] == 'xStar':
        _o_ = model._net(_i)
        loss_dict = {'loss': (_o - _o_).square().sum()}
        return combine_losses(loss_dict, coefs), loss_dict