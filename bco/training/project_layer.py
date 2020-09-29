import torch.nn.functional as F
import torch 
import numpy as np


class ProjectLayer(torch.nn.Linear):
    def __init__(self, 
                in_features=1, 
                n_planes=1,
                bias=True):
        self.project = True
        super(ProjectLayer, self).__init__(in_features, n_planes, bias=bias)
        self.out_features = in_features
        self.N  = 1000

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_()
        self.project_weights()

    def forward(self, x):
        u = self.weight
        b = self.bias

        val = x @ u.T - b
        absVal = val.abs()
        std = absVal.std(unbiased=False) + 1e-3
        mean = absVal.mean()
        softmin = torch.nn.functional.softmin(self.N * (absVal - mean)/std, dim=1)
        return x - (softmin @ u) * ((softmin * val).sum(1, keepdim=True))

    def project_weights(self):
        if not self.project:
            return
        with torch.no_grad():
            self.weight.data /= self.weight.data.norm(dim=1, keepdim=True)
