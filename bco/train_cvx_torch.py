from functools import reduce
import numpy as np
from functools import partial
import cvxpy as cp
import matplotlib.pyplot as plt
from ipdb import set_trace as st
import pandas as pd
import time
import torch
# torch.set_default_dtype(torch.double)
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from bco.training.datasets import process_data
from itertools import combinations
import os
from shutil import rmtree

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=False, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)

def random_signed_patterns(X_f, X_i, X_s, M, fuse_xxstar=False, device='cpu'):
    d = X_f.shape[1]
    W = torch.randn((M, d), device=device)
    w = 2*torch.rand((M, 1), device=device) - 1
    if fuse_xxstar:
        D_i = torch.sign(X_i @ W.T + w.T)
        D_s = D_i.clone()
        # D_s = torch.sign(X_s @ W.T + w.T)
        # keep = (D_i == D_s).all(axis=0) # Remove surfaces that separate Xi and Xs
        # D_i = D_i[:, keep]
        # D_s = D_s[:, keep]
        # W = W[keep]
        D_f = torch.sign(X_f @ W.T + w.T)

        # separate pairs of infeasible points
        W_ = torch.tensor(((X_i + X_s)[None, :, :] - (X_i + X_s)[:, None, :])/2, device=device).reshape((-1,d))
        D_f_ = torch.sign(X_f @ W_.T + w.T)
        D_i_ = torch.sign(X_i @ W_.T + w.T)
        D_s_ = torch.sign(X_s @ W_.T + w.T)
        
        D_f = torch.cat((D_f, D_f_), axis=1)
        D_i = torch.cat((D_i, D_i_), axis=1)
        D_s = torch.cat((D_s, D_s_), axis=1)
        W = torch.cat((W, W_), axis=1)
        _, idx = unique(torch.cat((D_f, D_i), dim=0), dim=1)
        # _, idx_= np.unique(torch.cat((D_f, D_i), 
        #                                 axis=0).numpy(), axis=1, return_index=True)
        
        # separate 
        
        # _, idx = np.unique(torch.cat((D_f, D_i), axis=0).numpy(), axis=1, return_index=True)
    else:  
        D_f = torch.sign(X_f @ W.T + w.T)
        D_i = torch.sign(X_i @ W.T + w.T)
        D_s = torch.sign(X_s @ W.T + w.T)
        _, idx = unique(torch.cat((D_f, D_i, D_s), dim=0), dim=1)
        # _, idx_= np.unique(torch.cat((D_f, D_i, D_s), axis=0).numpy(), axis=1, return_index=True)
    D_f = D_f[:, idx].contiguous()
    D_i = D_i[:, idx].contiguous()
    D_s = D_s[:, idx].contiguous()
    W = W[idx]
    w = w[idx]
    U = torch.cat((W,w), dim=1).T
    if 'cuda' in U.device.type:
        torch.cuda.empty_cache()
    return D_f, D_i, D_s, U

def enumerate_signed_patterns(X_f, X_i, X_s, fuse_xxstar, tol=1e-8):
    n_f, n_i, n_s = X_f.shape[0], X_i.shape[0], X_s.shape[0]
    n = n_f + n_i if fuse_xxstar else n_f + n_i + n_s
    D = np.ones((n, 2**n), dtype=np.int64)
    ite = 0
    for i in range(n+1):
        for idx in combinations(np.arange(n), r=i):
            D[idx, ite] = -1
            ite +=1
    if fuse_xxstar:
        D_f, D_i, D_s = D[:n_f], D[-n_i:], D[-n_i:]
    else:
        D_f, D_i, D_s = D[:n_f], D[n_f:n_f+n_i], D[-n_i:]

    # remove infeasible patterns:
    d = X_i.shape[1]
    U = cp.Variable((d, 2**n))
    s = cp.Variable((1,2**n), nonneg=True)
    con = [
        cp.multiply(D_i,(X_i @ U)) + s >= 0,
        cp.multiply(D_s,(X_s @ U)) + s>= 0
    ]
    if X_f.shape[0] > 0:
        con += [cp.multiply(D_f,(X_f @ U)) + s >= 0]
    prob = cp.Problem(cp.Minimize(cp.sum(s)), con)
    prob.solve()
    mask = (s.value <= tol).squeeze()
    return D_f[:, mask], D_i[:, mask], D_s[:, mask]

def plot_hyperplanes(X_f, X_i, X_s, h=None, M=None, show=True):
    x1, x2 = (torch.linspace(-1,1,100), torch.linspace(-1,1,100))
    f, a = plt.subplots(figsize=(5,4))
    
    if h is None:
        h = 2*torch.randn((M, X_f.shape[1]+1)) - 1

    a.plot(x1, ((h[:, :1] * x1 + h[:, -1:]) / h[:, 1:2]).T, linewidth=1)

    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

    # Plot data
    a.plot(X_f[:, 0], X_f[:, 1], 'g+')
    for i in range(X_i.shape[0]):
        a.plot([X_i[i,0], X_s[i,0]], [X_i[i,1], X_s[i,1]], '--+r')
    a.plot(X_s[:, 0], X_s[:, 1], 'g+')
    a.set_ylim([-1,1])
    a.set_xlim([-1,1])
    if show:
        plt.show()

def plot_model(model, hyperplanes=True, x_star=None, show=True, bounds_mul=1.):
    model.to('cpu')
    x1, x2 = torch.meshgrid(torch.linspace(-1,1,100), torch.linspace(-1,1,100))
    x = torch.stack((x1.flatten(), x2.flatten())).T *bounds_mul
    f, ax = plt.subplots(1, 2, figsize=(10,5))

    levels = np.linspace(-1,1.5,26)
    
    a = ax[0]
    da = ax[1]

    x.requires_grad = True
    with torch.no_grad():
        y, dy = model.value_and_gradient(x)
        y = y.reshape(x1.shape).detach()
        dy = dy.norm(dim=1).reshape(x1.shape).detach()
    c_ = a.contourf(x1, x2, (y), levels=levels, alpha=.5)
    c = a.contour(x1, x2, (y), levels=levels)
    co = f.colorbar(c, ax=a)
    c.collections[10].set_color('black')
    co.set_label('$score$', rotation=90)

    c_ = da.contourf(x1, x2, (dy), levels=30, alpha=.5)
    c = da.contour(x1, x2, (dy), levels=30)
    co = f.colorbar(c_, ax=da)
    co.set_label('$gradnorm$', rotation=90)

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

        # Plot data
        X_f, X_i, X_s = model.X_f, model.X_i, model.X_s
        a.plot(X_f[:, 0], X_f[:, 1], 'g+')
        for i in range(X_i.shape[0]):
            a.plot([X_i[i,0], X_s[i,0]], [X_i[i,1], X_s[i,1]], '--+r')
        a.plot(X_s[:, 0], X_s[:, 1], 'g+')

        # Plot x_star
        if x_star is not None:
            a.plot(x_star.T[0], x_star.T[1], 'ro')

        # Plot hyperplanes
        if hyperplanes:
            h = torch.cat((u1, u2),1).T
            x1, x2 = torch.linspace(-1,1,100), torch.linspace(-1,1,100)
            a.plot(x1, (-(h[:, :1] * x1 + h[:, -1:]) / h[:, 1:2]).T, linewidth=1)
    
        a.set_ylim([-1,1])
        a.set_xlim([-1,1])
        a.set_aspect('equal', 'box')
    ax[0].set_title('$f(x)$')
    ax[1].set_title('$\\Vert \\nabla f(x)\\Vert_2$')
    plt.tight_layout()
    if show:
        plt.show()
    return f

class CvxModel(torch.nn.Module):
    def __init__(self, abs_act=False, device='cpu',
                    data=None, state_dict=None, M=int(1e6), fuse_xxstar=False):
        super().__init__()
        assert data is None or state_dict is None
        self.abs_act = abs_act
        self.fuse_xxstar = fuse_xxstar
        self.M = M
        self.device = device

    # def train_prep(self, data=None, state_dict=False, M=int(1e6), fuse_xxstar=False):
        if data is not None:
            X_f, X_i, X_s, Y_i, dY_i = data
            X_f, X_i, X_s, Y_i, dY_i = X_f.to(device), X_i.to(device), X_s.to(device), Y_i.to(device), dY_i.to(device)
            D_f, D_i, D_s, u = random_signed_patterns(X_f, X_i, X_s, M, fuse_xxstar=fuse_xxstar, device=self.device)
            d = max([X_f.size(1), X_i.size(1)]) + 1
            u0, u1, u2 = torch.zeros((d, 1), device=self.device), u, u

        elif state_dict is not None:
            X_f, X_i, X_s, Y_i, dY_i = state_dict['X_f'], state_dict['X_i'], state_dict['X_s'], state_dict['Y_i'], state_dict['dY_i']
            D_f, D_i, D_s = state_dict['D_f'], state_dict['D_i'], state_dict['D_s']
            u0, u1, u2 = state_dict['u0'], state_dict['u1'], state_dict['u2']
        
        else:
            raise KeyError
        
        self.D_f = torch.nn.Parameter(D_f, requires_grad=False)
        self.D_i = torch.nn.Parameter(D_i, requires_grad=False)
        self.D_s = torch.nn.Parameter(D_s, requires_grad=False)
        self.X_f = torch.nn.Parameter(X_f, requires_grad=False)
        self.X_i = torch.nn.Parameter(X_i, requires_grad=False)
        self.X_s = torch.nn.Parameter(X_s, requires_grad=False)
        self.Y_i = torch.nn.Parameter(Y_i, requires_grad=False)
        self.dY_i = torch.nn.Parameter(dY_i, requires_grad=False)
    
        self.u0 = torch.nn.Parameter(u0, requires_grad=True)
        self.u1 = torch.nn.Parameter(u1, requires_grad=True)
        self.u2 = torch.nn.Parameter(u2, requires_grad=True)
        self.to(device)

    def value_and_gradient(self, X):
        id1 = (X @ self.u1[:-1]) + self.u1[-1]
        id2 = (X @ self.u2[:-1]) + self.u2[-1]
        if self.abs_act:
            id1 = torch.sign(id1)
            id2 = torch.sign(id2)
        else:
            id1 = id1.gt(0)*1.
            id2 = id2.gt(0)*1.
        # dy (X @ self.u0[:-1]) + self.u0[-1] + (id1 * sc1).sum(1, keepdims=True) - (id2 * sc2).sum(1, keepdims=True)
        st()
        dy = self.u0.T + id1 @ self.u1.T - id2 @ self.u2.T
        y = (X * dy[:, :-1]).sum(1) + dy[:, -1]
        return y, dy[:, :-1]

    def train_eval(self):
        if self.abs_act:
            dy_f = (self.D_f @ (self.u1 - self.u2).T) + self.u0.T
            dy_i = (self.D_i @ (self.u1 - self.u2).T) + self.u0.T
            dy_s = (self.D_s @ (self.u1 - self.u2).T) + self.u0.T
        else:
            dy_f = ((self.D_f + 1) @ (self.u1 - self.u2).T)/2 + self.u0.T
            dy_i = ((self.D_i + 1) @ (self.u1 - self.u2).T)/2 + self.u0.T
            dy_s = ((self.D_s + 1) @ (self.u1 - self.u2).T)/2 + self.u0.T

        y_f = (self.X_f * dy_f[:, :-1]).sum(1) + dy_f[:, -1]
        y_i = (self.X_i * dy_i[:, :-1]).sum(1) + dy_i[:, -1]
        y_s = (self.X_s * dy_s[:, :-1]).sum(1) + dy_s[:, -1]
        return y_f, y_i, y_s, dy_f[:, :-1], dy_i[:, :-1], dy_s[:, :-1]

    def con_eval(self):
        con_f_1 = - self.D_f * (self.X_f @ self.u1[:-1] + self.u1[-1])
        con_i_1 = - self.D_i * (self.X_i @ self.u1[:-1] + self.u1[-1])
        con_s_1 = - self.D_s * (self.X_s @ self.u1[:-1] + self.u1[-1])
        con_f_2 = - self.D_f * (self.X_f @ self.u2[:-1] + self.u2[-1])
        con_i_2 = - self.D_i * (self.X_i @ self.u2[:-1] + self.u2[-1])
        con_s_2 = - self.D_s * (self.X_s @ self.u2[:-1] + self.u2[-1])

        return con_f_1, con_i_1, con_s_1, con_f_2, con_i_2, con_s_2


    def sparsify(self, tol=1e-8):
        self.u1.data = self.u1[:, self.u1.norm(dim=0)>tol].contiguous()
        self.u2.data = self.u2[:, self.u2.norm(dim=0)>tol].contiguous()

    def loss(self, beta=1e-3, rho=1):
        loss_dict = {}
        y_f, y_i, y_s, dy_f, dy_i, dy_s = self.train_eval()
        loss_dict['loss_f'] = torch.relu(y_f).mean()
        loss_dict['loss_i'] = (y_i - self.Y_i.squeeze()).square().mean()
        loss_dict['loss_s'] = y_s.square().mean()
        loss_dict['loss_di'] = (dy_i - self.dY_i).square().mean()
        loss_dict['loss_ds'] = (dy_s - self.dY_i).square().mean()

        con_f_1, con_i_1, con_s_1, con_f_2, con_i_2, con_s_2 = self.con_eval()
        loss_dict['constraint'] = rho * (con_f_1.relu().square().sum() + con_i_1.relu().square().sum() + con_s_1.relu().square().sum() + con_f_2.relu().square().sum() + con_i_2.relu().square().sum() + con_s_2.relu().square().sum())

        loss_dict['group_norm'] = beta/2 * (self.u0.norm()/2 + self.u1.norm(dim=0).mean() + self.u2.norm(dim=0).mean())
        loss_dict['lipsh_norm'] = beta * (self.u0[:-1].norm()/2 + self.u1[:-1].norm(dim=0).mean() + self.u2[:-1].norm(dim=0).mean())
        loss = sum(loss_dict.values())
        return loss, loss_dict

    def freeze(self):
        pass

# def train_cvx(data=None, state_dict=None, n_epochs=1000,
#                 load_model=False, lr=1e-3, rho=1e-6, beta=1e-3, M=20000, 
#                 abs_act=False, tol=1e-8, beta=1e-4, device='cpu', nesting=0):
    
#     # Optimizer
#     opt = torch.optim.LBFGS(model.parameters(), lr=lr)

#     # Load state dict
#     if load_model:
#         model.load_state_dict(torch.load(model_name))
#         opt.load_state_dict(torch.load('opt_' + model_name))

#     return model, opt

if __name__ == "__main__":
    # filename = 'data/pacman/pacman_10_n3.csv'
    # input_variables = ['x1', 'x2']
    filename = 'data/multio_50/multio_2d_n4.csv'
    input_variables = ['x0', 'x1']
    abs_act = False
    device = 'cuda'
    rho = 1.
    beta = 1e-3
    M = int(1e6)
    model_name = 'multio_cuda'
    restart = False
    lr = 1e-3
    n_epochs = 10000

    # Load data and model
    if restart:
        model = CvxModel(state_dict = torch.load('models/'+model_name), abs_act=abs_act, device=device, M=M, fuse_xxstar = False)
    else:
        data = process_data(filename, 
                            output_regex='^sqJ$', 
                            input_columns=[v for v in input_variables])
        data = (data[0]/2, data[1]/2, data[2]/2, data[3]/2, data[4])
        model = CvxModel(data=data, abs_act=abs_act, device=device, M=M, fuse_xxstar = False)

    # Creat opt and writer
    opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=1000)
    log_dir=f'runs/{model_name}'
    if restart:
        opt.load_state_dict(torch.load('models/opt_'+ model_name))
    else:
        if os.path.exists(log_dir):
            rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Run training loop
    for e in range(n_epochs):
        # def closure():
        opt.zero_grad()
        loss, _ = model.loss()
        loss.backward()
            # return loss
        # opt.step(closure)
        opt.step()
        scheduler.step(loss)

        if e%100==0:
            with torch.no_grad():
                loss, loss_dict = model.loss(rho=rho, beta=beta)
                for l in loss_dict:
                    writer.add_scalar('loss/'+l,loss_dict[l], e)
                writer.add_scalar('loss/loss', loss, e)
                writer.add_scalar('lr', opt.param_groups[0]['lr'] , e   )
    
            # Save
            torch.save(model.state_dict(), f'models/{model_name}')
            torch.save(opt.state_dict(), f'models/opt_{model_name}')

    model.to('cpu')
    model.sparsify(1e-6)
    st()

    f = plot_model(model, hyperplanes=False, show=False)
    f.savefig('f2d_beta_abs_fuse_xxstar.pdf', bbox_inches='tight')