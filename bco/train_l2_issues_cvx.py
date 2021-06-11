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
from torch.autograd import grad

from bco.training.datasets import process_data
from itertools import combinations

def net(X, u1, u2, abs_act=False):
    sc1 = (X @ u1[:-1]) + u1[-1]
    sc2 = (X @ u2[:-1]) + u2[-1]
    id1 = torch.sign(sc1) if abs_act else sc1.gt(0)*1.
    id2 = torch.sign(sc2) if abs_act else sc2.gt(0)*1.
    return (id1 * sc1).sum(1, keepdims=True) - (id2 * sc2).sum(1, keepdims=True)

def random_signed_patterns(X_f, X_i, X_s, M, fuse_xxstar=False):
    d = X_f.shape[1]
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
    return D_f, D_i, D_s

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

def train_network(X_f, X_i, X_s, Y_i, beta, M=int(5000), fuse_xxstar=False, abs_act=False, sdf=True, norm=2):
    n_f, d = X_f.shape
    n_i, d = X_i.shape
    X_f = np.append(X_f,np.ones((n_f,1)),axis=1)
    X_i = np.append(X_i,np.ones((n_i,1)),axis=1)
    X_s = np.append(X_s,np.ones((n_i,1)),axis=1)
    d += 1
    Y_i = np.atleast_1d(Y_i.squeeze())

    dual_norm = np.inf if norm==1 else int(1/(1-1/float(norm)))

    ## Finite approximation of all possible sign patterns
    t0 = time.time()
    if n_f + 2*n_i < 15:
        D_f, D_i, D_s = enumerate_signed_patterns(X_f, X_i, X_s, fuse_xxstar)
    else:
        D_f, D_i, D_s = random_signed_patterns(X_f, X_i, X_s, M, fuse_xxstar=fuse_xxstar)
    m1 = D_f.shape[1]
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
        constraints += [
        ux_f_1>=0,
        ux_f_2>=0
        ]

        Y_f_lb = np.max(Y_i[:, None] - np.linalg.norm(X_f[None, :, :] - X_i[:, None, :], axis=-1, 
                                    ord=dual_norm), axis=0)
        if sdf:
            loss_f = cp.sum(cp.abs(y_f - Y_f_lb))
        else:
            loss_f = cp.sum(cp.pos(y_f))
        loss = loss + loss_f
        lipsh_f = cp.sum(cp.neg(y_f - Y_f_lb))
    else:
        loss_f = cp.Variable(value=0.)
        lipsh_f = cp.Variable(value=0.)

    # Infeasible points
    if n_i > 0:
        ux_i_1 = cp.multiply(D_i,(X_i @ Uopt1))
        ux_s_1 = cp.multiply(D_s,(X_s @ Uopt1))
        ux_i_2 = cp.multiply(D_i,(X_i @ Uopt2))
        ux_s_2 = cp.multiply(D_s,(X_s @ Uopt2))
    
        if abs_act:
            y_i = cp.sum(ux_i_1 - ux_i_2,axis=1)
            y_s = cp.sum(ux_s_1 - ux_s_2,axis=1)
            if sdf:
                if norm == 2:
                    constraints += [cp.sum(cp.square((Uopt1[:2] - Uopt2[:2]) @ D_i.T), axis=0) <= 1]
                elif norm == 1:
                    constraints += [cp.max(cp.abs((Uopt1[:2] - Uopt2[:2]) @ D_i.T), axis=0) <= 1]
        else:
            y_i = cp.sum(cp.multiply((D_i+1)/2,(X_i @ (Uopt1 - Uopt2))),axis=1)
            y_s = cp.sum(cp.multiply((D_s+1)/2,(X_s @ (Uopt1 - Uopt2))),axis=1)
            if sdf:
                if norm == 2:
                    constraints += [cp.sum(cp.square((Uopt1[:2] - Uopt2[:2]) @ ((D_i+1)/2).T), axis=0) <= 1]
                elif norm == 1:
                    constraints += [cp.max(cp.abs((Uopt1[:2] - Uopt2[:2]) @ ((D_i+1)/2).T), axis=0) <= 1]

        constraints += [
            ux_i_1>=0,
            ux_s_1>=0,
            ux_i_2>=0,
            ux_s_2>=0,
        ]
        loss_i = cp.sum(cp.abs(Y_i - y_i))
        loss_s = cp.sum(cp.abs(y_s))
        loss = loss + loss_i + loss_s
        lipsh_i = cp.sum(cp.neg(Y_i - y_i))
    else:
        loss_i = cp.Variable(value=0.)
        loss_s = cp.Variable(value=0.)
        lipsh_i = cp.Variable(value=0.)
        lipsh_s = cp.Variable(value=0.)

    # Regularization
    if norm == 2:
        reg = cp.mixed_norm(Uopt1[:-1].T,2,1) + cp.mixed_norm(Uopt2[:-1].T,2,1)
    elif norm == 1:
        reg = cp.Variable(nonneg=True)
        for i in range(d-1):
            for s in [-1,1]:
                constraints += [cp.sum(cp.pos(Uopt1[i] * s) + cp.pos(-Uopt2[i] * s)) <= reg]

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
        f'lipsh_f: {lipsh_f.value :.2E}, \n '
        f'lipsh_i: {lipsh_i.value :.2E}, \n '
        f'Time: {time.time()-t0 :.2f}s')
    if prob.status.lower() == 'infeasible':
        st()
        return None
    u1, u2 = torch.tensor(Uopt1.value), torch.tensor(Uopt2.value)
    st()

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

def plot_model(X_f, X_i, X_s, u1, u2, hyperplanes=True, x_star=None, abs_act=False, show=True, norm=2):
    x1, x2 = torch.meshgrid(torch.linspace(-1,1,100), torch.linspace(-1,1,100))
    x = torch.stack((x1.flatten(), x2.flatten())).T
    f, ax = plt.subplots(1, 2, figsize=(10,5))
    
    a = ax[0]
    da = ax[1]
    dual_norm = '\\infinity' if norm==1 else int(1/(1-1/float(norm)))
    if u1 is not None and u2 is not None:
        x.requires_grad = True
        y = torch.reshape(net(x, u1, u2, abs_act=abs_act), x1.shape)
        if type(dual_norm) == str and dual_norm == '\\infinity':
            dy = (grad(y.sum(), [x])[0]).max(dim=1)[0]
        else:
            dy = (grad(y.sum(), [x])[0]).norm(dual_norm, dim=1)
        dy = torch.reshape(dy, x1.shape)
        y, dy = y.detach(), dy.detach()
        c_ = a.contourf(x1, x2, (y), levels=30, alpha=.5)
        c = a.contour(x1, x2, (y), levels=30)
        a.contour(x1, x2, (y), levels=[-10, 0, 10], colors='k')
        co = f.colorbar(c, ax=a)
        co.set_label('$score$', rotation=90)

        c_ = da.contourf(x1, x2, (dy), levels=30, alpha=.5)
        c = da.contour(x1, x2, (dy), levels=30)
        co = f.colorbar(c_, ax=da)
        co.set_label('$gradnorm$', rotation=90)

    for a in ax:
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
        a.set_aspect('equal', 'box')
    ax[0].set_title('$f(x)$')
    ax[1].set_title('$\\Vert \\nabla f(x)\\Vert_{' + f"{dual_norm}" + "}$")
    plt.tight_layout()
    if show:
        plt.show()
    return f

class CvxModel(torch.nn.Module):
    def __init__(self, u1, u2, mean=0., std=1., abs_act=False):
        super().__init__()
        u1 = u1 if type(u1) == torch.Tensor else torch.tensor(u1)
        u2 = u2 if type(u2) == torch.Tensor else torch.tensor(u2)
        mean = mean if type(mean) == torch.Tensor else torch.tensor(mean)
        std = std if type(std) == torch.Tensor else torch.tensor(std)
        self.u1 = torch.nn.Parameter(u1, requires_grad=False)
        self.u2 = torch.nn.Parameter(u2, requires_grad=False)
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)
        self.abs_act = abs_act


    def forward(self, X):
        return net((X - self.mean)/self.std, self.u1, self.u2, abs_act=self.abs_act) * self.std
    
    def freeze(self):
        pass

def train_cvx(filename, input_variables, M=20000, abs_act=False, tol=1e-8, beta=1e-4, nesting=0):
    # Load data
    (fs, ifs, ifs_star, out, _) = process_data(filename, 
                                output_regex='^sqJ$', 
                                input_columns=[v for v in input_variables])

    if ifs.shape[0] ==0:
        u1 = torch.zeros((3,1))
        u2 = torch.zeros((3,1))
        u2[-1] = 1. # output will be -1 for every x -> always feasible
        return CvxModel(u1, u2, abs_act=False)

    # Scale
    n = (fs.shape[0] + 2*ifs.shape[0])
    mean = (ifs.sum(0, keepdims=True) + ifs_star.sum(0, keepdims=True) + fs.sum(0, keepdims=True)) / n
    std = (((ifs - mean).square().sum(0, keepdims=True) + 
            (ifs_star - mean).square().sum(0, keepdims=True) +
            (fs - mean).square().sum(0, keepdims=True)) / (n-1)).sqrt()
    std = std.mean() # to maintain the 1-Lishitzity I need to scale all dimensions by the same number.
    u1, u2 = train_network((fs - mean)/std, 
                            (ifs - mean)/std, 
                            (ifs_star - mean)/std, 
                            out / std, abs_act=abs_act, beta=beta)
    
    # Sparsify
    # u1 = u1[:, u1.norm(dim=0)>tol]
    # u2 = u2[:, u2.norm(dim=0)>tol]
    
    # Create model
    model = CvxModel(u1, u2, abs_act=abs_act, mean=mean, std=std)
    loss = ((model(ifs).squeeze() - out.squeeze()).abs().sum() + 
            model(ifs_star).abs().sum() + model(fs).relu().sum()) / n

    if loss >= 1e-6:
        if nesting <= 5:
            st()
            print(f"Loss is too high when training on {filename}. Increasing M, lower beta, and retrying.")
            return train_cvx(filename, input_variables, M=2*M, beta=beta/2, abs_act=abs_act, tol=tol, nesting=nesting+1)
        else:
            raise RecursionError("Maximum recursion depth for failed training reached.")
    return model

if __name__ == "__main__":
    data_file = 'data/twins/twins_10_n0.csv'
    xcols = ['x1', 'x2']
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
    
    abs_act = False
    norm=2

    torch.manual_seed(2)
    from data.trivial_shapes import square, pacman
    X_f, X_i, X_s, Y_i = square(6, norm=norm)

    X_f = X_f[:0]
    X_i = X_i[:2]
    X_s = X_s[:2]
    Y_i = Y_i[:2]
    st()
    u1, u2 = train_network(X_f, X_i, X_s, Y_i, abs_act=abs_act, 
                        fuse_xxstar=False, beta=1e-3, M=int(1e5), sdf=True, norm=norm)
    # u1 = u1[:, u1.norm(dim=0)>1e-6]
    # u2 = u2[:, u2.norm(dim=0)>1e-6]

    f = plot_model(X_f, X_i, X_s, u1, u2, hyperplanes=False,  abs_act=abs_act, show=False, norm=norm)
    plt.show()
    f.savefig('f2d_beta_abs_fuse_xxstar.pdf', bbox_inches='tight')