import torch
import numpy as np
from torch.autograd import grad
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F

def parse_coefs(params, device='cpu'):
    if params['model_type'] in ['sqJ_classifier_w_derivative', 'sqJ_orth_cert']:
        return {k: torch.tensor(params[kp], device=device) for k, kp in \
                zip(['gradient norm', 'sdf regularization', 'boundary regularization'], \
                    ["grad_norm_regularizer", "sdf_regularizer", "boundary_regularizer"]) if kp in params}
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

def controlled_leaky_relu(i, o, bound):
    # leaky relu unless o becomes so negative that it doesn't make sense for the bounds 7
    negative_slope = 0.01
    d0 = (i - bound[:1]).square().sum(dim=1, keepdim=True)
    d1 = (i - bound[1:]).square().sum(dim=1, keepdim=True)
    d = np.minimum(d1,d0).sqrt()
    return F.leaky_relu(o, negative_slope=negative_slope) + negative_slope * F.relu(-(o + d))

def loss_calc(batch, anchors, model, params, coefs={}):
    """Computes loss depending on modeltype

    Parameters
    ----------
    batch : list of tensors
        fl : torch.tensor
            tensor of feasible points
        ifl : torch.tensor
            tensor of infeasible points
        ifl_star : torch.tensor
            tensor of infeasible points projection
        o : torch.tensor
            tensor of infeasible points output
        do : torch.tensor
            tensor of infeasible points output derivative
    anchors : torch.tensor
        tensor of anchors for various regularizations
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
    (fs, ifs, ifs_star, out, dout) = batch
    _fs = model.normalize(input = fs)
    _ifs = model.normalize(input = ifs)
    _ifs_star = model.normalize(input = ifs_star)
    _o = model.normalize(output = out)
    _do = model.normalize(deriv = dout)
    if params['model_type'] == 'sqJ_classifier_w_derivative':

        # Infeasible points (regression)
        _ifs.requires_grad = True
        _o_ = model._net(_ifs)
        _do_ = grad(_o_.sum(), [_ifs], create_graph=True)[0]
        
        jpred = (_o - _o_).abs().mean()
        djpred = (_do - _do_).abs().sum(1, keepdim=True).mean()
        # import ipdb; ipdb.set_trace()
        
        # Projection of infeasible points (regression)
        _ifs_star.requires_grad = True
        _o_ = model._net(_ifs_star)
        _do_ = grad(_o_.sum(), [_ifs_star], create_graph=True)[0]
        
        jpred = jpred + _o_.abs().mean()
        djpred = djpred + (_do - _do_).abs().sum(1, keepdim=True).mean()
        
        # Feasible points (classification + grad norm reg)
        _fs.requires_grad = True
        _o_ = model._net(_fs)
        _do_ = grad(_o_.sum(), [_fs], create_graph=True)[0]
        feasible_class = F.relu(_o_).mean() * (2 * (fs.size(1) + 1))# 2x(dim+1) to account for xStar and derivatives
        grad_norm = ((1 - torch.square(_do_ / (model.input_std/model.output_std)).sum(dim=1)).abs().mean()
                    #  + model._net(_fs - _do_ * _o_).abs().mean()
                    )

        # Regularization anchors
        if anchors is not None and params['sdf_regularizer'] > 0:
            anchors.requires_grad = True
            anchout = model._net(anchors, reuse=True)
            grd = grad(anchout.sum(), [anchors], create_graph=True)[0]
            anchors.requires_grad = False
            sdf_reg = ((1 - torch.square(grd / (model.input_std/model.output_std)).sum(dim=1)).abs().mean() 
                        #  + (model._net(anchors - grd * anchout)).abs().mean()
                        )
        else:
            sdf_reg = torch.zeros(1)
        # Infeasible boundary
        boundary_reg = torch.zeros(1)
        if anchors is not None and params['boundary_regularizer'] > 0:
            bound  = torch.tensor(params['bounds'])
            for b in range(2):
                for j in range(_ifs.size(1)):
                    boundary_reg = boundary_reg + F.relu(-model._net(anchors.index_fill(1, torch.tensor(j), bound[b][j]), reuse=True)).mean()
                    # a = anchors.index_fill(1, torch.tensor(j), bound[b][j])
                    # a.requires_grad = True
                    # m = model._net(a, reuse=True)
                    # g = grad(m.sum(), [a], create_graph=True)[0]
                    # boundary_reg = boundary_reg + F.relu(-m).mean() # + (1-g.square().sum()).abs().mean()
                    # import ipdb; ipdb.set_trace()
                    
        loss_dict = {'J loss':jpred, 
                    'dJ loss': djpred, 
                    'classification loss': feasible_class, 
                    'gradient norm': grad_norm,
                    'sdf regularization': sdf_reg,
                    'boundary regularization': boundary_reg,
                    }
        loss = combine_losses(loss_dict, coefs)

        return loss, loss_dict

    else: 
        # Backward compatibility hack (ok becaus this part of the code will be rarely used)
        input = torch.cat((fs, ifs))
        
        classes = torch.zeros(input.size(0))
        classes[:fl.shape[0]] = 1

        output = torch.zeros(input.size(0))
        output[fl.shape[0]:] = out

        dout = torch.zeros((input.size(0), 2*input.size(1)+1))
        dout[fl.shape[0]:] = dout

        # /!\ does not contain projected data ifs_Star
        
        _i = model.normalize(input = i)
        _o = model.normalize(output = o)
        _do = model.normalize(deriv = do)
        if params['model_type'] == 'sqJ_hinge_classifier':
            _i.requires_grad = True
            if _i.grad is not None:
                _i.grad.detach_()
                _i.grad.zero_()

            _o_ = model._net(_i)
            _do_ = grad(_o_.sum(), [_i], create_graph=True)[0]
            _i.requires_grad = False
            grad_norm = torch.abs(torch.square(_do_ / (model.input_std/model.output_std)).sum(dim=1) - 1.).sum()

            loss_dict = {
                        'gradient norm': grad_norm
                        }
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

            feasible = torch.abs(cl)
            N_feasible = np.maximum(feasible.sum(), 1)
            infeasible = (1 - torch.abs(cl))
            N_infeasible = np.maximum(feasible.sum(), 1)

            jpred = ((_o - _o_).abs().T @ infeasible) / N_infeasible
            djpred = ((_do - _do_).abs().sum(1, keepdim=True).T @ infeasible) / N_infeasible
            feasible_class =  (F.relu(_o_ * cl).T @ feasible) / N_feasible * (2 * (_i.size(1) + 1))# 2x(dim+1) to account for xStar and derivatives
            grad_norm = (torch.abs(torch.norm(_do_ / (model.input_std/model.output_std), dim=1) - 1.).T @ feasible) / N_feasible
            certif_norm = torch.norm(certif, dim=1).mean()

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
            loss_dict = {'loss': ((_o - _o_).square() + (_do - _do_).square().sum(1, keepdim=True)).sum()}
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = torch.linspace(-2, .1, 100).unsqueeze(1)
    bounds = torch.tensor([[-1.], [1.]])
    i = torch.tensor([[0.]])
    
    l = controlled_leaky_relu(i, t, bounds)
    plt.plot(t, l)
    plt.show()
