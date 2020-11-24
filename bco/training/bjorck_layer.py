# Only slightly modified from https://github.com/cemanil/LNets/blob/master/lnets/models/layers/dense/bjorck_linear.py
# and from https://github.com/cemanil/LNets/blob/master/lnets/utils/math/projections/l2_ball.py
# Ref Paper:  https://arxiv.org/abs/1811.05381, Cem Anil, James Lucas, Roger Grosse


import torch.nn.functional as F
import torch 
import numpy as np

class NrmLinear(torch.nn.Linear):
    def __init__(self, 
                in_features=1, 
                out_features=1,
                bias=True):
        super(NrmLinear, self).__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        # torch.nn.init.orthogonal_(self.weight, gain=stdv)
        torch.nn.init.orthogonal_(self.weight, gain=1.)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-1., 1.)

    def forward(self, x):
        out = F.linear(x, self.weight)
        out = out.div(out.norm()) * x.norm()
        return out + self.bias

class BjorckLinear(torch.nn.Linear):
    def __init__(self, 
                in_features=1, 
                out_features=1, 
                bjorck_beta=0.5,
                bjorck_iter=20,
                bjorck_order=1,
                safe_scaling = True,
                bias=True):

        self.bjorck_beta = bjorck_beta
        self.bjorck_iter = bjorck_iter
        self.bjorck_order = bjorck_order
        self.safe_scaling = safe_scaling
        self.ortho_w = None
        super(BjorckLinear, self).__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        # torch.nn.init.orthogonal_(self.weight, gain=stdv)
        torch.nn.init.orthogonal_(self.weight, gain=1.)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-1., 1.)

    def forward(self, x, reuse=False):
        if not reuse:
            self.ortho_w = self.orthonormalize()
        assert not torch.isnan(self.ortho_w).any(), "Bjorck Orthonormalization did not converge"
        return F.linear(x, self.ortho_w, self.bias)

    def orthonormalize(self, safe_scaling=None, bjorck_beta=None, bjorck_iter=None, bjorck_order=None):
        safe_scaling = safe_scaling if safe_scaling else self.safe_scaling
        bjorck_beta = bjorck_beta if bjorck_beta else self.bjorck_beta
        bjorck_iter = bjorck_iter if bjorck_iter else self.bjorck_iter
        bjorck_order = bjorck_order if bjorck_order else self.bjorck_order

        # Scale the values of the matrix to make sure the singular values are less than or equal to 1.
        if safe_scaling:
            scaling = get_safe_bjorck_scaling(self.weight)
        else:
            scaling = 1

        return bjorck_orthonormalize(self.weight.t() / scaling,
                                        beta =  bjorck_beta,
                                        iters = bjorck_iter,
                                        order = bjorck_order).t()


    def project_weights(self, safe_scaling, bjorck_beta, bjorck_iter, bjorck_order):
        with torch.no_grad():
            self.weight.data.copy_(self.orthonormalize(safe_scaling, bjorck_beta, bjorck_iter, bjorck_order))

    def freeze(self):
        self.project_weights(self.safe_scaling, 
                            self.bjorck_beta, 
                            self.bjorck_iter,
                            self.bjorck_order)
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.__class__ = torch.nn.Linear



def bjorck_orthonormalize(w, beta=0.5, iters=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """
    # TODO: Make sure the higher order terms can be implemented more efficiently.
    if order == 1:
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w = (1 + beta) * w - beta * w.mm(w_t_w)

    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w = (+ (15 / 8) * w
                 - (5 / 4) * w.mm(w_t_w)
                 + (3 / 8) * w.mm(w_t_w_w_t_w))

    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)

            w = (+ (35 / 16) * w
                 - (35 / 16) * w.mm(w_t_w)
                 + (21 / 16) * w.mm(w_t_w_w_t_w)
                 - (5 / 16) * w.mm(w_t_w_w_t_w_w_t_w))

    elif order == 4:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)

        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w_t_w_w_t_w = w_t_w.mm(w_t_w)
            w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w)
            w_t_w_w_t_w_w_t_w_w_t_w = w_t_w.mm(w_t_w_w_t_w_w_t_w)

            w = (+ (315 / 128) * w
                 - (105 / 32) * w.mm(w_t_w)
                 + (189 / 64) * w.mm(w_t_w_w_t_w)
                 - (45 / 32) * w.mm(w_t_w_w_t_w_w_t_w)
                 + (35 / 128) * w.mm(w_t_w_w_t_w_w_t_w_w_t_w))

    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w


def get_safe_bjorck_scaling(weight):
    return torch.tensor([np.sqrt(weight.shape[0] * weight.shape[1])]).float().to(weight.device)