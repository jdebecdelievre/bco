import os
import shutil
import json
import torch 
from torch.autograd import grad
from bco.training.models import build_model, scalarize
from bco.training.datasets import build_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import golden
import re 
import argparse
GREEN = '\033[92m'
RED = '\033[91m'
ENDCOL = '\033[0m'


def apply(fn, last=False):
    files = [f[:-5] for f in os.listdir('.') if f.endswith('.json') and 'sqrt' not in f]
    files.sort(key=lambda f: os.path.getctime(f+'.json'))
    if last:
        files = [files[-1]]
    for f in files:

        try:
            with open(f + '.json', 'r') as ff:
                content = json.load(ff)
        except:
            print("Failed opening ", f)
            continue

        content, delete = fn(f, content)

        if delete:
            pass
            remove(f)
        else:
            with open(f + '.json', 'w') as ff:
                json.dump(content, ff, indent=4)
    return 


def remove(f):
    files = os.listdir('.')

    for f_ in filter(lambda x: re.match(f+'.+', x), files):
        try:
            os.remove(f_)
            # print(f_)
        except OSError:
            print('Error removing ' + f+'.json')
            pass
    

def filter_test(filename, content):
    if ('loss' not in content):
        print(f'Removing {filename}: no loss recorded')
        return content, True
    if 'epochs' in content and content['epochs'] < 1000:
        print(f'Removing {filename}: Only {content["epochs"]} epochs')
        return content, True
    return content, False


def correct_train_data_size(filename, content):
    if 'twins' in filename:
        casename = 'twins'
    elif 'disk' in filename:
        casename = 'disk'
    content['train_set_size'] = int(re.findall(r'%s(\d+)' % (casename +'_'), filename)[0])
    return content, False


def compute_Jloss(filename, content, plot_contours=True):
    if 'Jloss' in content and 'dJloss' in content and 'test_accuracy' in content:
        return content, False

    model = build_model(content)
    try:
        model.load_state_dict(torch.load(filename+'.mdl'))
    except RuntimeError:
        return content, False
    model.eval()
    dataset, test_dataset = build_dataset(content, test=True)
    test_input, test_output, _, test_classes = test_dataset.get_dataset()
    inp, out, _, classes = dataset.get_dataset()
    with torch.no_grad():
        content.update(model.metrics(test_input, test_output, test_classes, prefix= 'test_'))
        content.update(model.metrics(inp, out, classes, prefix= 'train_'))

    # Evaluate the number of points that should not have been missed
    with torch.no_grad():
        # select infeasible train points
        inp = inp[(1-classes).squeeze().bool()].unsqueeze(0)
        out = out[(1-classes).squeeze().bool()]

        # select misclassified test points
        pred_classes = model.classify(test_input)
        misclassified = pred_classes != test_classes
        assert misclassified.sum() == content['test_false_positives'] + content['test_false_negatives']
        misclassified_ex = test_input[misclassified].unsqueeze(1)

    # compute the distance between misclassified tests and infeasible train
    dist = (inp - misclassified_ex).norm(dim=-1)
    easy = (dist <= out[:, 0]).any(0)
    content['avoidable_misclassified'] = int(scalarize(easy.sum()))
    print(f'There were {easy.sum().numpy()} avoidable misclassifications')


    # Find optimal tolerance
    if content['model_type'] in ['xStar', 'sqJ']:
        inp, out, _, classes = dataset.get_dataset()
        def acc(tol):
            with torch.no_grad():
                p = model.classification_metrics(model.classify(inp, tol=tol), classes, prefix='')
            return -p['accuracy']
        tol_opt = golden(acc, brack=[1e-4, 1.], full_output=True)[0]
        with torch.no_grad():
            content.update(model.metrics(test_input, test_output, test_classes, prefix= 'test_'))
            content.update(model.metrics(inp, out, classes, prefix= 'train_'))
        content['tol_opt'] = tol_opt

    if plot_contours and content['model']['input_size'] == 2:
        bounds=None if 'bounds' not in content else content['bounds']
        f = score_contours(model, dataset, save=filename + 'contours', 
                            grad_norm=False,bounds=bounds)
        plt.close()
        f = score_contours(model, save=filename + 'grad_norm', grad_norm=True, bounds=bounds)
        plt.close()

        sl = 0. if 'tol_opt' not in content else content['tol_opt']
        f = score_contours(model, dataset, single_level=sl, save=filename + 'decision_boundary', bounds=bounds)
        plt.close()

        f = loss_map(model, dataset, filename+'_lossmap', bounds=bounds)
        plt.close()

    if content['model_type'] == 'sqJ_orth_cert':
        x1, x2 = torch.meshgrid(torch.linspace(-2,2), torch.linspace(-2,2))
        X = torch.stack((x1.flatten(), x2.flatten())).T
        with torch.no_grad():
            Y = model._net.certificate(X)
            Y = Y.norm(dim=1)
            Y = Y.reshape(x1.shape)
        f, a = plt.subplots(figsize=(5,4))
        c = a.contourf(x1, x2, (Y), levels=30)
        a.axis('equal')
        inp, out, _, cls = dataset.get_dataset()
        a.scatter(inp[:,0], inp[:,1], s=10., c=-cls[:,0], marker='+')
        try:
            co = f.colorbar(c, ax=a)
            co.set_label('$UQ$', rotation=90)
        except:
            pass
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        f.savefig(filename + 'UQ.pdf', bbox_inches='tight')
    return content, False

def loss_map(model, test_dataset, save, bounds=None):
    f, a = plt.subplots(figsize=(5,4))

    bounds = bounds if bounds is not None else [[-2,-2], [2,2]]
    (fs, ifs, ifs_star, out, dout) = test_dataset.tensors
    
    # Feasible points
    a.scatter(fs[:,0], fs[:,1], s=20., c='green', marker='+')
    cls = model.classify(fs)
    a.scatter(fs[:,0], fs[:,1], s=20., c=cls, marker='o')
    
    for i in range(ifs.size(0)):
        a.plot([ifs[i,0], ifs_star[i,0]], [ifs[i,1], ifs_star[i,1]], '--+r')

    # Infeasible points
    ifs.requires_grad = True
    _o_ = model._net(ifs)
    jpred = (out - _o_).abs().detach()
    ifs.requires_grad = False
    # import ipdb; ipdb.set_trace()
    a.scatter(ifs[:,0], ifs[:,1], s=20., c=jpred.log().squeeze(), marker='o')
    for i in range(ifs.size(0)):
        a.annotate('{0:.2e}'.format(jpred.squeeze()[i]), 
                    (ifs[i,0], ifs[i,1]),
                    fontsize='xx-small')

    # Projected points
    ifs_star.requires_grad = True
    _o_ = model._net(ifs_star)
    jpred = (_o_).abs().detach()
    ifs_star.requires_grad = False
    a.scatter(ifs_star[:,0], ifs_star[:,1], s=20., c=jpred.log().squeeze(), marker='+')
    for i in range(ifs.size(0)):
        a.annotate('{0:.2e}'.format(jpred.squeeze()[i]), 
        (ifs_star[i,0], ifs_star[i,1]),
        fontsize='xx-small')

    a.set_xlim([bounds[0][0], bounds[1][0]])
    a.set_ylim([bounds[0][1], bounds[1][1]])
    f.savefig(save+'.pdf', bbox_inches='tight')

def score_contours(model, test_dataset=None, single_level=None, save=None, 
                bounds=None, grad_norm=False):
    bounds = bounds if bounds is not None else [[-2,-2], [2,2]]
    x1, x2 = torch.meshgrid(torch.linspace(bounds[0][0], bounds[1][0]),
                            torch.linspace(bounds[0][1], bounds[1][1]))
    X = torch.stack((x1.flatten(), x2.flatten())).T

    if not grad_norm:
        with torch.no_grad():
            if not hasattr(model, 'output_mean') or model.output_mean.shape[0] == 1:
                Y = model(X)
            elif model.output_mean.shape[0] > 1 :
                Y = model.predict_sqJ(X)
    else:
        X.requires_grad = True
        if X.grad is not None:
            X.grad.detach_()
            X.grad.zero_()
        if not hasattr(model, 'output_mean') or model.output_mean.shape[0] == 1:
            Y = model(X)
        elif model.output_mean.shape[0] > 1 :
            Y = model.predict_sqJ(X)
        dY = grad(Y.sum(), [X], create_graph=False)[0]
        X.requires_grad = False
        Y = torch.norm(dY, dim=1, keepdim=True)
        # Y = torch.sigmoid(-Y * torch.norm(dY, dim=1, keepdim=True) / (1 - torch.norm(dY, dim=1, keepdim=True))).detach()
    Y = Y.reshape(x1.shape)

    f, a = plt.subplots(figsize=(5,4))
    if single_level is None:
        c_ = a.contourf(x1, x2, (Y), levels=30, alpha=.5)
        c = a.contour(x1, x2, (Y), levels=30)
    else:
        c = a.contourf(x1, x2, (Y), levels=np.array([single_level, 1e10]))
        
    if test_dataset is not None:
        (fs, ifs, ifs_star, _, _) = test_dataset.tensors
        a.scatter(fs[:,0], fs[:,1], s=20., c='green', marker='+')
        for i in range(ifs.size(0)):
            a.plot([ifs[i,0], ifs_star[i,0]], [ifs[i,1], ifs_star[i,1]], '--+r')

    a.axis('equal')
    if single_level is None:
        try:
            co = f.colorbar(c, ax=a)
            co.set_label('$score$', rotation=90)
        except:
            pass
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

    if save is not None:
        f.savefig(save+'.pdf', bbox_inches='tight')
    return f


def match_runs_models():    
    model_files_json = [f[:-5] for f in os.listdir('.') if f.endswith('.json')]
    model_files_mdl = [f[:-4] for f in os.listdir('.') if f.endswith('.mdl')]
    # run_files =  [f for f in os.listdir('runs')]
    run_files= []

    # Delete json files that don't have an MDL file
    for f in model_files_json:
        if f not in model_files_mdl:
            print(f + RED +  ' not matched ' + ENDCOL)
            remove(f)
        else:
            print(f + GREEN + ' matched ' + ENDCOL)

    # Delete mdl files that don't have a json file
    for f in model_files_mdl:
        if f not in model_files_json:
            print(f + RED +  ' not matched' + ENDCOL)
            remove(f)
        else:
            print(f + GREEN + ' matched' + ENDCOL)
    
    # Delete run files that don't correspond to a valid json
    for f in run_files:
        if f not in model_files_json:
            print(f + RED + ' not matched' + ENDCOL)
            remove(f)
        else:
            print(f + GREEN + ' matched' + ENDCOL)


def extract_dataFrame():
    files = [f[:-5] for f in os.listdir('.') if f.endswith('.json')]
    C = []
    for f in files:
        try:
            with open(f + '.json', 'r') as ff:
                C.append(json.load(ff))
        except:
            print("Failed opening ", f)
            continue
        C[-1]['name'] = f
    C = pd.DataFrame(C)
    C.to_csv('summary.csv', index=False)


def accuracy_plot(folders, labels=None):
    data = {}
    for f in folders:
        D = pd.read_csv(f + f'/summary.csv')
        D = D.select_dtypes(exclude=['O', 'bool'])
        data[f] = D

    from matplotlib.ticker import MaxNLocator

    f, a = plt.subplots(figsize=(6,6))

    for f in data:
        d = data[f]
        dd = d.groupby('input_size', as_index=None)
        dm = dd.mean()
        ds = dd.std()
        # a.plot(dm.input_size, dm.test_accuracy, '+-')
        a.errorbar(dm.input_size, dm.test_accuracy, 
                    yerr=ds.test_accuracy,
                    capsize=4)
    labels = labels if labels else folders
    a.legend(labels)
    a.grid(True)
    a.set_xlabel('Input Space Dimension')
    a.set_ylabel('Accuracy near Boundary')
    a.set_ylim([30,100])
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    plt.savefig('accuracy.pdf', bbox_inches='tight')


parser = argparse.ArgumentParser(description='PostProc Options')
parser.add_argument('action', metavar='N', type=str,
                    help='action to perform', choices=['jloss', 'extractdf', 'acc'])
parser.add_argument('-l', '--last', action='store_true')



if __name__ == "__main__":
    args = parser.parse_args()
    # match_runs_models()

    # Delete if missing saved items
    # apply(filter_test)
    
    # Correct train data size
    # apply(correct_train_data_size)

    # Compute Jloss
    if args.action == 'jloss':
        apply(compute_Jloss, last = args.last)

    # Extract pd
    if args.action == 'extractdf':
        extract_dataFrame()

    # Create summary accuracy plot (call outside folder)
    if args.action == 'acc':
        pts = [10, 30, 50, 150]
        accuracy_plot([f'models_ball{i}_L1' for i in pts],
                            labels = [f'{i} data points' for i in pts])

