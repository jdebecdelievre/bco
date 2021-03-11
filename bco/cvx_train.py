import numpy as np
# from jax import np
# from jax import partial, jit
from functools import partial
import cvxpy as cp
import matplotlib.pyplot as plt
import ipdb
import pandas as pd
import time

# RELU
def act(x):
    return np.maximum(0,x)
def d_act(x):
    return x>=0

# ## ABS
# def act(x):
#     return np.abs(x)
# def d_act(x):
#     return np.sign(x)

def train_sqJ(X, y, beta, M=int(5000), dY=None):
    n, d = X.shape
    X = np.append(X,np.ones((n,1)),axis=1)
    d += 1

    ## Finite approximation of all possible sign patterns
    t0 = time.time()
    dmat=np.empty((n,M))
    u = np.empty((d, M))
    for i in range(M):
        u[:, i] = np.random.randn(d)
        dmat[:, i] = np.dot(X,u[:, i])
    indic = 2 * (dmat >= 0) - 1
    indic, idx = (np.unique(indic,axis=1, return_index=True))
    u = u[:, idx]
    dmat = d_act(dmat[:, idx]) 
    print(f'Dmat creation: {time.time()-t0}s')

    # Optimal CVX
    m1=dmat.shape[1]
    Uopt1=cp.Variable((d,m1), value=np.random.randn(d,m1))
    Uopt2=cp.Variable((d,m1), value=np.random.randn(d,m1))

    yopt1=cp.sum(cp.multiply(dmat,(X @ Uopt1)),axis=1)
    yopt2=cp.sum(cp.multiply(dmat,(X @ Uopt2)),axis=1)

    cost1 = cp.sum_squares(y - (yopt1-yopt2))/2/n
    cost2 = (cp.sum(cp.norm(Uopt1, 2, 0)) + cp.sum(cp.norm(Uopt2, 2, 0)))/n

    constraints = [
        cp.multiply(indic,(X @ Uopt1))>=0,
        cp.multiply(indic,(X @ Uopt2))>=0,
        cp.square(y - (yopt1-yopt2)) <= 1e-5, # fit
        # cp.sum(cp.norm((Uopt1-Uopt2)[:-1], 2, 0)) <= 1 # lipshitz
    ]

    if dY is not None:
        for n_ in range(n):
            # I could loop over dimensions instead of datapoints - but will make removing feasible points slightly more difficult
            dy = cp.sum(cp.multiply(dmat[n_:n_+1], Uopt1[:-1] - Uopt2[:-1]), axis=1)
            constraints.append(cp.sum_squares(dy - dY[n_]) <= 1e-6)

    prob=cp.Problem(cp.Minimize(cost1  + beta * cost2),constraints)
    options = dict(mosek_params = {'MSK_DPAR_BASIS_TOL_X':1e-8})
    t0 = time.time()
    prob.solve(solver=cp.MOSEK, **options)
    # prob.solve(solver=cp.SCS)
    print(f'Status: {prob.status}, Value: {prob.value}, Time: {time.time()-t0}s')
    if prob.status.lower() == 'infeasible':
        ipdb.set_trace()
        return None

    # Weights
    u1_nrm = np.linalg.norm(Uopt1.value, axis=0, keepdims=True)
    u2_nrm = np.linalg.norm(Uopt2.value, axis=0, keepdims=True)

    idx = np.logical_and((u1_nrm[0] >= 1e-6), (u2_nrm[0]>=1e-6))
    u1 = Uopt1.value[:, idx]
    u2 = Uopt2.value[:, idx]
    u1_nrm = u1_nrm[:, idx]
    u2_nrm = u2_nrm[:, idx]
    W = np.hstack((u1 / np.sqrt(u1_nrm), u2 / np.sqrt(u2_nrm))).T
    wf = np.hstack((np.sqrt(u1_nrm), - np.sqrt(u2_nrm)))
    
    v_w = u1 - u2

    return (W, wf, u, v_w)
    # return partial(ycalc, W=W, wf=wf)
    # return partial(model, u=u, v_w=Uopt1.value - Uopt2.value, deriv=True)

def ycalc(x, W=0., wf=0., deriv=False):
    # use act
    if x.shape[1] != W.shape[1]:
        x = np.append(x, np.ones((x.shape[0],1)),axis=1)
    if deriv:
        return np.squeeze(act(x @ W.T) @ wf.T), (d_act(x @ W.T) * wf) @ W
    else:
        return np.squeeze(act(x @ W.T) @ wf.T)

def dat_dy(dmat, u1, u2):
    dy = np.zeros((dmat.shape[0], u1.shape[0]-1))
    for n_ in range(dy.shape[0]):
        dy[n_] = np.sum(dmat[n_] * (u1[0]- u2[0]))
    return dy

def model(x, u=0., v_w=0., deriv=False):
    # Use indicator functions
    if x.shape[1] != u.shape[0]:
        x = np.append(x, np.ones((x.shape[0],1)),axis=1)
    assert u.shape == v_w.shape
    assert x.shape[1] == u.shape[0]

    indic = d_act(x @ u)
    if deriv:
        return (indic * (x @ (v_w))).sum(1), indic @ ((v_w)).T
    else:
        return (indic * (x @ (v_w))).sum(1)

def plot_model(X, y, classes, weights, Xstar=None, indic=False):
    (W, wf, u, v_w) = weights
    x1, x2 = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
    x = np.stack((x1.flatten(), x2.flatten())).T
    if indic:
        y = np.reshape(model(x, u, v_w), x1.shape)
    else:
        y = np.reshape(ycalc(x, W, wf), x1.shape)

    f, a = plt.subplots(figsize=(5,4))
    c_ = a.contourf(x1, x2, (y), levels=30, alpha=.5)
    c = a.contour(x1, x2, (y), levels=30)
    a.contour(x1, x2, (y), levels=[-10, 0, 10], colors='k')
    co = f.colorbar(c, ax=a)
    co.set_label('$score$', rotation=90)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    if Xstar is not None:
        for i in range(Xstar.shape[0]):
            a.plot([X[i,0], Xstar[i,0]], [X[i,1], Xstar[i,1]], '--+', 
                    color='green' if classes[i] > 1/2 else 'red')
    else:
        a.scatter(X[:,0], X[:,1], s=20., c=classes, marker='+')
    plt.show()

def train_sqJ_cls(X, y, cl, beta=1e-4, M=int(3000), dY=None, Xstar=None):
    n, d = X.shape
    X = np.append(X,np.ones((n,1)),axis=1)
    d += 1

    ## Finite approximation of all possible sign patterns
    t0 = time.time()
    dmat=np.empty((n,M))
    u = np.empty((d, M))
    for i in range(M):
        u[:, i] = np.random.randn(d)
        dmat[:, i] = np.dot(X,u[:, i])
    indic = 2 * (dmat >= 0) - 1
    indic, idx = (np.unique(indic,axis=1, return_index=True))
    u = u[:, idx]
    dmat = d_act(dmat[:, idx]) 
    print(f'Dmat creation: {time.time()-t0}s. {dmat.shape[1]} sign patterns.')

    # Optimal CVX
    m1=dmat.shape[1]
    Uopt1=cp.Variable((d,m1), value=np.random.randn(d,m1))
    Uopt2=cp.Variable((d,m1), value=np.random.randn(d,m1))

    yopt1=cp.sum(cp.multiply(dmat,(X @ Uopt1)),axis=1)
    yopt2=cp.sum(cp.multiply(dmat,(X @ Uopt2)),axis=1)

    cost1 = ((y - (yopt1-yopt2))**2) @ (1-cl)
    cost2 = cp.sum(cp.pos(cp.multiply(y, 2*cl-1)))
    cost3 = (cp.sum(cp.norm(Uopt1, 2, 0)) + cp.sum(cp.norm(Uopt2, 2, 0)))/n

    constraints = [
        cp.multiply(indic,(X @ Uopt1))>=0,
        cp.multiply(indic,(X @ Uopt2))>=0,
        # cp.multiply(cp.square(y - (yopt1-yopt2)), 1-cl) <= 1e-5, # fit
        # cp.multiply(y, 2*cl-1) <=0, # cls
        # cp.sum(cp.norm((Uopt1-Uopt2)[:-1], 2, 0)) <= 1 # lipshitz
    ]

    # Add derivative (applies both to x and xstar)
    if dY is not None:
        for i in range(d-1):
            dy = (cp.sum(cp.multiply(dmat, Uopt1[i:i+1] - Uopt2[i:i+1]), axis=1))
            # constraints.append(cp.multiply(cp.square(dy - dY[:, i]), 1-cl) <= 1e-6)
            cost = cp.square(dy - dY[:, i]) @ (1-cl)

    # Repeat for Xstar
    if Xstar is not None:
        Xstar = np.append(Xstar,np.ones((n,1)),axis=1)
        yopt1=cp.sum(cp.multiply(dmat,(Xstar @ Uopt1)),axis=1)
        yopt2=cp.sum(cp.multiply(dmat,(Xstar @ Uopt2)),axis=1)

        cost2 = cost2 + cp.square((yopt1-yopt2)) @ (1-cl)

        constraints = constraints + [
            cp.multiply(indic,(Xstar @ Uopt1)) >= 0,
            cp.multiply(indic,(Xstar @ Uopt2)) >= 0,
            # cp.multiply(cp.square((yopt1-yopt2)), 1-cl) <= 1e-5, # fit
        ]

    prob=cp.Problem(cp.Minimize(cost1 + cost2 + beta * cost3),constraints)
    options = dict(
        # mosek_params = {'MSK_DPAR_BASIS_TOL_X':1e-8}
        )
    t0 = time.time()
    prob.solve(solver=cp.MOSEK, **options)
    # prob.solve(solver=cp.SCS)
    print(f'Status: {prob.status}, Value: {prob.value}, Time: {time.time()-t0}s')
    if prob.status.lower() == 'infeasible':
        ipdb.set_trace()
        return None

    # Weights
    v_w = (Uopt1 - Uopt2).value
    u1_nrm = np.linalg.norm(Uopt1.value, axis=0, keepdims=True)
    u2_nrm = np.linalg.norm(Uopt2.value, axis=0, keepdims=True)

    idx = (u1_nrm[0] >= 1e-10)
    u1 = Uopt1.value[:, idx]
    u1_nrm = u1_nrm[:, idx]
    idx = (u2_nrm[0] >= 1e-10)
    u2 = Uopt2.value[:, idx]
    u2_nrm = u2_nrm[:, idx]
    W = np.hstack((u1 / np.sqrt(u1_nrm), u2 / np.sqrt(u2_nrm))).T
    wf = np.hstack((np.sqrt(u1_nrm), - np.sqrt(u2_nrm)))

    # ipdb.set_trace()
    return (W, wf, u, v_w)

if __name__ == "__main__":
    # data_file = 'data/twins/twins_30_n0.csv'
    # xcols = ['x1', 'x2']
    data_file = 'data/multio_50/multio_2d_n4.csv'
    xcols = ['x0', 'x1']
    augment = True

    data = pd.read_csv(data_file)
    X = data[xcols].values
    Y = data['sqJ'].values
    dY = data[['dsqJ_'+k for k in xcols]].values
    classes = data['classes'].values
    if augment:
        Xstar = data[[k + '_star' for k in xcols]].values
        X = np.append(X, Xstar, axis=0)
        classes = np.append(classes, classes, axis=0)
        Y = np.append(Y, Y*0, axis=0)
        dY = np.append(dY, dY, axis=0)
    else:
        Xstar = None
    # (W, wf, u, v_w) = train_sqJ(X, Y, 1e-4, dY=dY)
    weights = train_sqJ_cls(X, Y, classes, M=10000, dY=dY, Xstar=None)
    plot_model(X, Y, classes, weights, Xstar=Xstar, indic=True)