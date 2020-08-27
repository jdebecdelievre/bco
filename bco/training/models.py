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

from lnets.models.activations import GroupSort
from bco.training.bjorck_layer import BjorckLinear

ACTIVATIONS = {
    'relu':nn.ReLU,
    'tanh':nn.Tanh,
    'logsigmoid':nn.LogSigmoid,
    'identity':nn.Identity,
    'groupsort':lambda :GroupSort(1)
}

def get_activation(keyword):
    keyword = keyword.lower()
    if 'groupsort' in keyword and len(keyword) != len('groupsort'):
        groupsize = int(keyword.split('groupsort')[-1])
        return lambda :GroupSort(groupsize)
    else:
        return ACTIVATIONS[keyword]

from lnets.models.layers import BjorckLinear as BjorckLinear_

class DefConfig(dict):
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

config = DefConfig(cuda=False, 
        model=DefConfig(linear=DefConfig(safe_scaling=True, bjorck_beta=0.5, bjorck_iter=20, bjorck_order=1)))

def get_linear(options):
    if options['type'].lower() == 'linear':
        return torch.nn.Linear
    elif 'bjorck' in options['type'].lower():
        opt = options.copy()
        del opt['type']
        # return lambda inp, out: BjorckLinear_(inp, out, bias=True, config=config)
        return lambda inp, out: BjorckLinear(inp, out, **opt)
    else:
        raise NotImplementedError

def vectorize_params(value, n, get_fn = lambda x:x):
    if type(value) == list:
        assert len(value) == n
        valvec = []
        for i in range(n):
            valvec.append(get_fn(value[i]))
    else:
        valvec = [get_fn(value)] * n
    return valvec

def build_layers(hidden_layers, activation, linear_layers, input_size, output_size):
    n = len(hidden_layers)
    activation = vectorize_params(activation, n, get_activation)
    linear_layers = vectorize_params(linear_layers, n+1, get_linear)

    layers = [
        linear_layers[0](input_size, hidden_layers[0]),
        activation[0]() 
        ]

    for i in range(n-1):
        layers += [linear_layers[i](hidden_layers[i], hidden_layers[i+1]),
        activation[i+1]()
        ]
    model = nn.Sequential(*layers, linear_layers[i](hidden_layers[-1], output_size))

    print(model)
    return model


def scalarize(val):
    return np.atleast_1d(val.squeeze().detach().data.numpy())[0]

class sqJModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.params = model_params
        self._net = build_layers(
            model_params['hidden_layers'], 
            model_params['activation'],
            model_params['linear'], 
            model_params['input_size'], 1)
        self.input_mean = nn.Parameter(torch.zeros(model_params['input_size']) ,requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(model_params['input_size']) ,requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(1) ,requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(1) ,requires_grad=False)

    def normalize(self, input=None, output=None, deriv=None):
        assert sum([int(i is not None) for i in [input, output, deriv]]) == 1, "Only one tensor can be normalized by the method"
        if input is not None:
            _input = (input - self.input_mean)/ self.input_std
            return _input
        if output is not None:
            _output = (output - self.output_mean)/ self.output_std
            return _output
        if deriv is not None:
            _deriv = deriv / self.output_std * self.input_std
            return _deriv

    def unnormalize(self, _input=None, _output=None, _deriv=None):
        assert sum([int(i is not None) for i in [_input, _output, _deriv]]) == 1, "Only one tensor can be unnormalized by the method"
        if _input is not None:
            input = _input * self.input_std + self.input_mean
            return input
        if _output is not None:
            output = _output * self.output_std + self.output_mean
            return output
        if _deriv is not None:
            deriv = _deriv * self.output_std / self.input_std
            return deriv

    def forward(self, input):
        """Computes *unnormalized* model output

        Parameters
        ----------
        input : torch.tensor
            *unnormalized* input

        Returns
        -------
        output : torch.tensor
            *unnormalized* output
        """
        _input = self.normalize(input=input)
        return self.unnormalize(_output=self._net(_input))
    
    def _cut_off(self, _output, tol=0.):
        """Applies classification cutoff

        Parameters
        ----------
        _output : torch.tensor
            *normalized* _output of model._net.forward
        tol : float, optional
            tolerance cutoff, by default 0.

        Returns
        -------
        torch.tensor
            classes
        """
        return np.heaviside(-_output + tol, 1.)

    def classify(self, input, tol=0.):
        """Classifies *unnormalized* inputs

        Parameters
        ----------
        input : torch.tensor
            input
        tol : float, optional
            tolerance cutoff, by default 0.

        Returns
        -------
        torch.tensor
            classes
        """
        with torch.no_grad():
            _input = self.normalize(input=input)
            _output=self._net(_input)
            return self._cut_off(_output=_output, tol=tol)

    def classification_metrics(self, classes, true_classes, writer=None, e=None, prefix=''):
        """Logs classification Metrics

        Parameters
        ----------
        classes : torch.tensor
            Classes predicted by model
        true_classes : torch.tensor
             
        writer : tensorboard.SummaryWriter, optional
            Writer to log scalar metrics, by default None
        e : int, optional
            epoch to log on writer, by default None
        prefix : str, optional
            prefix to metrics logging, by default ''

        Returns
        -------
        dict
            Dictionary of metrics values
        """
        return classification_metrics(classes, true_classes, writer=writer, e=e, prefix=prefix)

    def predict_sqJ(self, input):
        """Predicts square root of J from *unnormalized* input

        Parameters
        ----------
        input : torch.tensor
            

        Returns
        -------
        torch.tensor
            sqJ
        """
        with torch.no_grad():
            sqJ = F.relu(self.forward(input))
        return sqJ

    def predict_J(self, input):
        return torch.square(self.predict_sqJ(input))

    def J_rmse(self, input, J_val):
        J = self.predict_J(input)
        return ((J - J_val)**2).mean().sqrt()

    def classes_and_J(self, input):
        """Predicts both class label and J value from *unnormalized* input

        Parameters
        ----------
        input : torch.tensor
            

        Returns
        -------
        classes : torch.tensor
        
        J : torch.tensor

        """
        with torch.no_grad():
            _input = self.normalize(input=input)
            _output=self._net(_input)
            classes = self._cut_off(_output=_output)
            output = self.unnormalize(_output=_output)
            J = F.relu(output)**2
        return classes, J

    def metrics(self, input, output, classes, writer=None, e=None, prefix=''):
        """Computes and logs all metrics : classification and J estimation

        Parameters
        ----------
        input : torch.tensor
            *unnormalized input*
        output : torch.tensor
            target output
        classes : torch.tensor
            target classes
        writer : tensorboard.SummaryWriter, optional
            Writer to log scalar metrics, by default None
        e : int, optional
            epoch to log on writer, by default None
        prefix : str, optional
            prefix to metrics logging, by default ''

        Returns
        -------
        dict
            Dictionary of metrics values
        """
        J = F.relu(output)**2
        classes_, J_ = self.classes_and_J(input)

        metrics = self.classification_metrics(classes_, classes, writer, e, prefix)

        Jrmse = ((J - J_)**2).mean().sqrt()
        metrics[prefix + 'Jrmse'] = scalarize(Jrmse)
        if writer is not None:
            assert e is not None, "Provide epoch number"
            writer.add_scalar(prefix + 'Jrmse', Jrmse, e)
        return metrics

    
    def extract_features(self, input):
        """Returns the output of the last hidden layer

        Parameters
        ----------
        input : torch.tensor
            unnormalized input, just like in forward

        Returns
        -------
        torch.tensor
            Output of the last hidden layer

        """
        with torch.no_grad():
            _x = self.normalize(input=input)
            for layer in self._net[:-1]:
                _x = layer(_x)
        return _x

    def projection(self, time):
        # time = 'epoch' or 'update'
        with torch.no_grad():
            if self.params[f'per_{time}_proj']['turned_on']:
                for layer in (self._net):
                    if type(layer) == BjorckLinear and hasattr(layer, 'weight'):
                        layer.project_weights(
                            bjorck_beta  = self.params[f'per_{time}_proj']["bjorck_beta"],
                            bjorck_iter  = self.params[f'per_{time}_proj']["bjorck_iter"],
                            bjorck_order = self.params[f'per_{time}_proj']["bjorck_order"],
                            safe_scaling = self.params[f'per_{time}_proj']["safe_scaling"]
                        )


class CompoundModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, input):
        return torch.max(torch.cat([m.forward(input) for m in self.models], dim=-1), dim=-1)[0]

class SubNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input):
        return self.net(input)[:,:1]

class OrthonormalCertificates(sqJModel):
    def __init__(self, model_params):
        super().__init__()
        self.params = model_params
        self._net_ = build_layers(
            model_params['hidden_layers'], 
            model_params['activation'],
            model_params['linear'], 
            model_params['input_size'], 1)
        self._net  = SubNet(self._net_)
        self.input_mean = nn.Parameter(torch.zeros(model_params['input_size']) ,requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(model_params['input_size']) ,requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(1) ,requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(1) ,requires_grad=False)

    def certificate(self, input):
        return self._net(input)[1:]
    
    # def __iter__(self) -> Iterator[torch.nn.Module]:
    #     return self.net.__iter__()


class xStarModel(sqJModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self._nets = torch.nn.ModuleList([build_layers(
            model_params['hidden_layers'], 
            model_params['activation'],
            model_params['linear'], 
            model_params['input_size'], 1) for i in range(model_params['input_size'])])
        self.input_mean = nn.Parameter(torch.zeros(model_params['input_size']), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(model_params['input_size']), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(model_params['input_size']), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(model_params['input_size']), requires_grad=False)

    def forward(self, input):
        _input = self.normalize(input=input)
        return torch.cat([self.unnormalize(_output=m(_input)) for m in self._nets], axis=1)
    
    def _cut_off(self, _output, tol=1e-1):
        output = _output * self.output_std + self.output_mean
        sqJ = torch.sqrt((output**2).sum(1))
        clas =  super()._cut_off(sqJ, tol)
        return clas

    def predict_sqJ(self, input):
        with torch.no_grad():
            output = self.forward(input)
        return torch.sqrt((output**2).sum(1))
    

def build_model(params):
    if type(params)== str:
        with open(params, 'r') as f:
            params = json.load(f)
    if params['model_type'] in ['sqJ_classifier_w_derivative', 'sqJ_classifier', 'sqJ', 'classifier']:
        return sqJModel(params['model'])
    elif params['model_type'] == 'xStar':
        return xStarModel(params['model']) 
    elif params['model_type'] == 'sqJ_orth_cert':
        return OrthonormalCertificates(params['model'])
    else:
        raise NotImplementedError


def classification_metrics(classes, true_classes, writer=None, e=None, prefix=''):
    """Logs classification Metrics

    Parameters
    ----------
    classes : torch.tensor
        Classes predicted by model
    true_classes : torch.tensor
            
    writer : tensorboard.SummaryWriter, optional
        Writer to log scalar metrics, by default None
    e : int, optional
        epoch to log on writer, by default None
    prefix : str, optional
        prefix to metrics logging, by default ''

    Returns
    -------
    dict
        Dictionary of metrics values
    """
    true_positives = (classes.T @ true_classes)
    true_negatives =  (1-classes).T @ (1-true_classes)
    false_positives = classes.T @ (1-true_classes)
    false_negatives =  (1-classes).T @ (true_classes)
    test_accuracy = (true_positives + true_negatives) / true_classes.shape[0] * 100.

    if (true_positives + false_positives) == 0:
        # print(f'Output is all 0 at iter {e}')
        test_precision = torch.zeros(1)
    else:
        test_precision = true_positives / (true_positives + false_positives) * 100.

    if (true_positives + false_negatives) == 0:
        test_recall = torch.zeros(1)
    else:
        test_recall = true_positives / (true_positives + false_negatives) * 100.

    if test_precision + test_recall == 0:
        f1 = -torch.ones(1)
    else:
        f1 = 2 * test_precision * test_recall / (test_precision + test_recall )  

    params = {}
    params[prefix + 'accuracy'] = scalarize(test_accuracy)
    params[prefix + 'precision'] = scalarize(test_precision)
    params[prefix + 'recall'] = scalarize(test_recall)
    params[prefix + 'true_positives'] = scalarize(true_positives)
    params[prefix + 'true_negatives'] = scalarize(true_negatives)
    params[prefix + 'false_positives'] = scalarize(false_positives)
    params[prefix + 'false_negatives'] = scalarize(false_negatives)
    params[prefix + 'f1'] = scalarize(f1)

    if writer is not None:
        assert e is not None, "Provide epoch number"
        writer.add_scalar('accuracy', test_accuracy, e)
        writer.add_scalar('precision', test_precision, e)
        writer.add_scalar('recall', test_recall, e)
        writer.add_scalar('true_positives', true_positives, e)
        writer.add_scalar('true_negatives', true_negatives, e)
        writer.add_scalar('false_positives', false_positives, e)
        writer.add_scalar('false_negatives', false_negatives,e)
        writer.add_scalar('f1', f1,e)

    return params
