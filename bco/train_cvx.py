from functools import reduce
import numpy as np
from functools import partial
import cvxpy as cp
import matplotlib.pyplot as plt
from ipdb import set_trace as st
import pandas as pd
import time
import torch
torch.set_default_dtype(torch.double)

from bco.training.datasets import process_data

def net(X, u1, u2, abs_act=False):
    sc1 = X @ u1
    sc2 = X @ u2
    id1 = torch.sign(sc1) if abs_act else sc1.gt(0)*1.
    id2 = torch.sign(sc2) if abs_act else sc2.gt(0)*1.
    return (id1 * sc1).sum(1) - (id2 * sc2).sum(1)
    # y = torch.zeros(X.shape[0])
    # for i, xx in enumerate(X):
    #     sc1 = xx @ u1
    #     sc2 = xx @ u2
    #     id1 = torch.sign(sc1) if abs_act else sc1.gt(0)*1.
    #     id2 = torch.sign(sc2) if abs_act else sc2.gt(0)*1.
    #     y[i] = id1 @ sc1 - id2 @ sc2
    # return y

def train_network(X_f, X_i, X_s, Y_i, beta, M=int(5000), fuse_xxstar=False, abs_act=False, sdf=False):
    n_f, d = X_f.shape
    n_i, d = X_i.shape
    X_f = np.append(X_f,np.ones((n_f,1)),axis=1)
    X_i = np.append(X_i,np.ones((n_i,1)),axis=1)
    X_s = np.append(X_s,np.ones((n_i,1)),axis=1)
    d += 1

    ## Finite approximation of all possible sign patterns
    t0 = time.time()
    W = 2*torch.randn((M, d)) - 1
    if fuse_xxstar:
        D_i = torch.sign((torch.tensor(X_i) @ W.T))
        D_s = D_i.clone()
        # D_s = torch.sign((torch.tensor(X_s) @ W.T))
        # keep = (D_i == D_s).all(axis=0) # Remove surfaces that separate Xi and Xs
        # D_i = D_i[:, keep]
        # D_s = D_s[:, keep]
        # W = W[keep]
        D_f = torch.sign((torch.tensor(X_f) @ W.T))

        # separate pairs of infeasible points
        W = torch.tensor(((X_i + X_s)[None, :, :] - (X_i + X_s)[:, None, :])/2).reshape((-1,d))
        D_f_ = torch.sign((torch.tensor(X_f) @ W.T))
        D_i_ = torch.sign((torch.tensor(X_i) @ W.T))
        D_s_ = torch.sign((torch.tensor(X_s) @ W.T))
        
        D_f = torch.cat((D_f, D_f_), axis=1)
        D_i = torch.cat((D_i, D_i_), axis=1)
        D_s = torch.cat((D_s, D_s_), axis=1)
        _, idx = np.unique(torch.cat((D_f, D_i), 
                                        axis=0).numpy(), axis=1, return_index=True)
        
        # separate 
        
        # _, idx = np.unique(torch.cat((D_f, D_i), axis=0).numpy(), axis=1, return_index=True)
    else:  
        D_f = torch.sign((torch.tensor(X_f) @ W.T))
        D_i = torch.sign((torch.tensor(X_i) @ W.T))
        D_s = torch.sign((torch.tensor(X_s) @ W.T))
        _, idx = np.unique(torch.cat((D_f, D_i, D_f), axis=0).numpy(), axis=1, return_index=True)
    D_f = D_f[:, idx]
    D_i = D_i[:, idx]
    D_s = D_s[:, idx]
    m1 = len(idx)
    print(f'Dmat creation: {time.time()-t0}s, {m1} arrangements identified.')

    # Optimal CVX
    Uopt1=cp.Variable((d,m1), value=np.random.randn(d,m1))
    Uopt2=cp.Variable((d,m1), value=np.random.randn(d,m1))
    constraints = []
    loss = 0

    # Feasible points
    if n_f > 0:
        ux_f_1 = cp.multiply(D_f,(X_f @ Uopt1))
        ux_f_2 = cp.multiply(D_f,(X_f @ Uopt2))
        y_f = cp.sum(ux_f_1 - ux_f_2,axis=1) if abs_act \
                 else cp.sum(cp.multiply((D_f+1)/2,(X_f @ (Uopt1 - Uopt2))),axis=1)
        # s_f = cp.Variable(n_f, value=np.random.rand(n_f), nonneg=True) # slack
        constraints += [
        ux_f_1>=0,
        ux_f_2>=0,
        # y_f + s_f == 0,
        ]
        st()
        Y_f_lb = np.max(Y_i[:, None] - np.linalg.norm(X_f[None, :, :] - X_i[:, None, :], axis=-1), axis=0)
        loss_f = cp.sum(cp.abs(y_f - Y_f_lb))
        loss = loss + loss_f
    else:
        loss_f = cp.Variable(value=0.)

    # Infeasible points
    if n_i > 0:
        ux_i_1 = cp.multiply(D_i,(X_i @ Uopt1))
        ux_s_1 = cp.multiply(D_s,(X_s @ Uopt1))
        ux_i_2 = cp.multiply(D_i,(X_i @ Uopt2))
        ux_s_2 = cp.multiply(D_s,(X_s @ Uopt2))
    
        if abs_act:
            y_i = cp.sum(ux_i_1 - ux_i_2,axis=1)
            y_s = cp.sum(ux_s_1 - ux_s_2,axis=1)
        else:
            y_i = cp.sum(cp.multiply((D_i+1)/2,(X_i @ (Uopt1 - Uopt2))),axis=1)
            y_s = cp.sum(cp.multiply((D_s+1)/2,(X_s @ (Uopt1 - Uopt2))),axis=1)
    
        # s_i = cp.Variable(n_i, value=np.random.rand(n_i), nonneg=True)
        # s_s = cp.Variable(n_i, value=np.random.rand(n_i), nonneg=True)
        constraints += [
            ux_i_1>=0,
            ux_s_1>=0,
            ux_i_2>=0,
            ux_s_2>=0,
            # y_i == s_i + Y_i,
            # y_s + s_s == 0,
        ]
        loss_i = cp.sum(cp.abs(Y_i - y_i))
        loss_s = cp.sum(cp.abs(y_s))
        loss = loss_i + loss_s
        lipsh_f = cp.sum(cp.neg(y_f - Y_f_lb))
        lipsh_i = cp.sum(cp.neg(Y_i - y_i))
    else:
        loss_i = cp.Variable(value=0.)
        loss_f = cp.Variable(value=0.)
        lipsh_i = cp.Variable(value=0.)
        lipsh_f = cp.Variable(value=0.)

    # Regularization
    reg = cp.mixed_norm(Uopt1.T,2,1) + cp.mixed_norm(Uopt2.T,2,1)

    # Solution
    prob=cp.Problem(cp.Minimize(100*(loss + beta * reg)),constraints)
    t0 = time.time()
    options = dict(mosek_params = {'MSK_DPAR_BASIS_TOL_X':1e-8})
    prob.solve(solver=cp.MOSEK, verbose=True, **options)
    # prob.solve(solver=cp.SCS)

    print(f'Status: {prob.status}, \n '
        f'Value: {prob.value :.2E}, \n '
        f'loss_f: {loss_f.value :.2E}, \n '
        f'loss_i: {loss_i.value :.2E}, \n '
        f'loss_s: {loss_s.value :.2E}, \n '
        f'Reg: {reg.value : .2E}, \n ' 
        f'lipsh_f: {lipsh_f :.2E}, \n '
        f'lipsh_i: {lipsh_i :.2E}, \n '
        f'Time: {time.time()-t0 :.2f}s')
    if prob.status.lower() == 'infeasible':
        st()
        return None
    u1, u2 = torch.tensor(Uopt1.value), torch.tensor(Uopt2.value)

    return u1, u2

def minimize(h, U, upper, lower, M=100):
    d = u.shape[0] - 1

    ## Finite approx_imation of all possible sign patterns
    t0 = time.time()
    X = torch.zeros((M, d+1))
    X[:, :2] = torch.rand((M, d))*(upper - lower) + lower
    D_ = torch.sign(X @ torch.tensor(h)).T
    D, idx = np.unique(D_.numpy(), axis=1, return_index=True)
    m1 = D.shape[1]
    print(f'Dmat creation: {time.time()-t0}s')

    def opt(d, u):
        x = cp.Variable(d)

        # Predictions
        ux = cp.multiply(d,(x @ u[:-1] + u[-1]))
        if abs_act:
            y = cp.sum(ux)
        else:
            y = cp.sum(cp.multiply((d+1)/2,(x @ u[:-1] + u[-1])))

        # Constraints
        constraints = [
            ux >= 0,
            x <= upper, x>= lower
        ]

        prob=cp.Problem(cp.Minimize(cp.minimum(y)),constraints)
        t0 = time.time()
        prob.solve()
        print(f'Status: {prob.status}, Value: {prob.value}, Time: {time.time()-t0}s')
        if prob.status.lower() == 'infeasible':
            return None
        return x.value, y.value
    
    X = np.zeros((d, m1))
    Y = np.zeros(m1)
    for i in range(m1):
        X[:, i], Y[i] = opt(D[:, i], U[:, i])
    i = np.argmin(Y[i])
    return X[:, i], Y[i]

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

def plot_model(X_f, X_i, X_s, u1, u2, hyperplanes=True, x_star=None, abs_act=False, show=True):
    x1, x2 = torch.meshgrid(torch.linspace(-1,1,100), torch.linspace(-1,1,100))
    x = torch.stack((x1.flatten(), x2.flatten(), torch.ones_like(x2.flatten()))).T
    f, a = plt.subplots(figsize=(5,4))
    
    if u1 is not None and u2 is not None:
        y = torch.reshape(net(x, u1, u2, abs_act=abs_act), x1.shape)
        c_ = a.contourf(x1, x2, (y), levels=30, alpha=.5)
        c = a.contour(x1, x2, (y), levels=30)
        a.contour(x1, x2, (y), levels=[-10, 0, 10], colors='k')
        co = f.colorbar(c, ax=a)
        co.set_label('$score$', rotation=90)
    
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

    # Plot data
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


    if show:
        plt.show()

class CvxModel(torch.nn.Module):
    def __init__(self, u1, u2, abs_act):
        self.u1 = torch.nn.Parameter(torch.tensor(u1), requires_grad=False)
        self.u2 = torch.nn.Parameter(torch.tensor(u2), requires_grad=False)
        self.abs_act = abs_act

    def forward(self, X):
        return net(X, u1, u2, abs_act=self.abs_act)

def train_cvx(filename, input_variables, M=20000, tol=1e-6):
    # Load data
    (fs, ifs, ifs_star, out, _) = process_data(filename, 
                                output_regex='^sqJ$', 
                                input_columns=[v for v in input_variables])
    n = fs.shape[0] + ifs.shape[0]
    u1, u2 = train_network(fs, ifs, ifs_star, out, abs_act=True, beta=1e-2)

    u1 = u1[:, u1.norm(dim=0)>tol]
    u2 = u2[:, u2.norm(dim=0)>tol]

    return CvxModel(u1, u2, act_abs=True)

if __name__ == "__main__":
    # data_file = 'data/twins/twins_50_n0.csv'
    # xcols = ['x1', 'x2']
    # data_file = 'data/disk/disk_10_n1.csv'
    # xcols = ['x1', 'x2']
    # data_file = 'data/multio_50/multio_2d_n4.csv'
    # xcols = ['x0', 'x1']

    # data = pd.read_csv(data_file)
    # classes = data.classes
    # data_f = data[classes == 1]
    # X_f = data_f[xcols].values / 2
    # data_i = data[classes == 0]
    # X_i = data_i[xcols].values / 2
    # X_s = data_i[[x + '_star' for x in xcols]].values / 2
    # Y_i = data_i['sqJ'].values / 2

    from data.trivial_shapes import square, pacman
    # torch.manual_seed(4)
    X_f, X_i, X_s, Y_i = pacman(50)

    abs_act = True
    u1, u2 = train_network(X_f, X_i, X_s, Y_i, abs_act=abs_act, fuse_xxstar=False, beta=1e-2, M=int(1e4), sdf=True)
    # u1, u2 = None, None
    u1 = u1[:, u1.norm(dim=0)>1e-6]
    u2 = u2[:, u2.norm(dim=0)>1e-6]
    h = torch.cat((u1, u2),1)
    # h = h / h.norm(0, keepdim=True)
    # plot_hyperplanes(X_f, X_i, X_s, h=h.T, show=False)
    plot_model(X_f, X_i, X_s, u1, u2, hyperplanes=False,  abs_act=abs_act, show=False)
    plt.show()