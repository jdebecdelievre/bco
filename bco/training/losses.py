import torch
import numpy as np
from torch.autograd import grad
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F

def parse_coefs(params, device='cpu'):
    if params['model_type'] in ['sqJ_classifier_w_derivative', 'sqJ_orth_cert']:
        return {k: F.relu(torch.tensor(params[kp], device=device)) for k, kp in \
                zip(['gradient norm', 'sdf regularization', 'boundary regularization', 'gradient', 'gradient'], \
                    ["grad_norm_regularizer", "sdf_regularizer", "boundary_regularizer", 'dJ loss', 'dJ loss z*']) if kp in params}
    else:
        return {}

def combine_losses(loss_dict, coefs):
    loss = 0
    for l in loss_dict:
        if l in coefs:
            if coefs[l] > 0:
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

def grad_norm_reg(i, o, do, model):
    return ((1 - torch.square(do / (model.input_std/model.output_std)).sum(dim=1)).abs().mean()
                #+ model._net(i - do * o).abs().mean()
            )

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
        
        loss_dict = {}
        loss_dict['gradient norm'] = torch.zeros(1)
        if _ifs.size(0) > 0:
            # Infeasible points (regression)
            _ifs.requires_grad = True
            _o_ = model._net(_ifs)
            _do_ = grad(_o_.sum(), [_ifs], create_graph=True)[0]

            loss_dict['J loss'] = (_o - _o_).abs().mean()
            loss_dict['dJ loss'] = _o.T @ ((_do - _do_).abs().mean(1, keepdim=True)) / _ifs.shape[0]
            # djpred = (_ifs - _o_ * _do_ - _ifs_star).abs().mean()
            loss_dict['gradient norm'] = loss_dict['gradient norm'] + grad_norm_reg(_ifs, _o_, _do_, model)
            
            # Projection of infeasible points (regression)
            _ifs_star.requires_grad = True
            _o_ = model._net(_ifs_star, reuse=True) 
            _do_ = grad(_o_.sum(), [_ifs_star], create_graph=True)[0]
            
            loss_dict['J loss z*'] = _o_.abs().mean()
            loss_dict['dJ loss z*'] = _o.T @ ((_do - _do_).abs().mean(1, keepdim=True)) / _ifs.shape[0]
            # djpred = djpred + ((_do - _do_).abs().sum(1, keepdim=True)* _o).mean()
            loss_dict['gradient norm'] = loss_dict['gradient norm'] + grad_norm_reg(_ifs_star, _o_, _do_, model)

        if _fs.size(0) > 0:
            # Feasible points (classification + grad norm reg)
            _fs.requires_grad = True
            _o_ = model._net(_fs, reuse=True)
            _do_ = grad(_o_.sum(), [_fs], create_graph=True)[0]
            
            loss_dict['classification loss']= F.relu(_o_).mean() * (2 * (fs.size(1) + 1))# 2x(dim+1) to account for xStar and derivatives
            loss_dict['gradient norm'] = loss_dict['gradient norm'] + grad_norm_reg(_fs, _o_, _do_, model)

        # Regularization anchors
        if anchors is not None:
            _anchors = model.normalize(input = anchors)
            # Sdf property
            if params['sdf_regularizer'] > 0:
                _anchors.requires_grad = True
                anchout = model._net(_anchors, reuse=True)
                grd = grad(anchout.sum(), [_anchors], create_graph=True)[0]
                _anchors.requires_grad = False
                loss_dict['sdf_regularizer'] = grad_norm_reg(_anchors, anchout, grd, model)

            # Monotonicity
            boundary_reg = torch.zeros(1)
            if "monotonicity" in params:
                mono_reg = torch.zeros(1)
                for i,m in enumerate(params["monotonicity"]):
                    if m is None:
                        continue
                    else:
                        mono_reg = mono_reg + F.relu(- m * grd[:, i]).mean()
                loss_dict['monotonicity regularization'] = mono_reg  

            # Infeasible boundary
            if params['boundary_regularizer'] > 0:
                boundary_reg = torch.zeros(1)
                bound  = torch.tensor(params['bounds'])
                for b in range(2):
                    for j in range(_ifs.size(1)):
                        boundary_reg = boundary_reg + F.relu(-model._net(anchors.index_fill(1, torch.tensor(j), bound[b][j]), reuse=True)).mean()
                loss_dict['boundary_regularizer'] = boundary_reg
        loss = combine_losses(loss_dict, coefs)
        return loss, loss_dict

    elif params['model_type'] == 'sqJ_hinge_classifier':
        loss_dict = {}
        if _ifs.size(0) > 0:
            # Infeasible points (regression)
            _o_ = model._net(_ifs)
            # _o_, _do_ = model._net.get_value_and_gradient(_ifs)
            loss_dict['J loss'] = F.relu(_o - _o_).mean()
            # loss_dict['inf_grd'] = (_do_.square().sum(1) - 1).square().mean()
            
            # Projection of infeasible points (regression)
            _o_ = model._net(_ifs_star) 
            # _o_, _do_ = model._net.get_value_and_gradient(_ifs_star)
            loss_dict['J loss z*'] = F.relu(_o_).mean()
            # loss_dict['inf_star_grd'] = (_do_.square().sum(1) - 1).square().mean()


        if _fs.size(0) > 0:
            # Feasible points (classification + grad norm reg)
            _o_ = model._net(_fs)
            # _o_, _do_ = model._net.get_value_and_gradient(_fs)
            loss_dict['classification loss']= F.leaky_relu(_o_).mean()
            # loss_dict['gradient norm'] = _do_.square().sum(dim=1)
            # loss_dict['fs_star_grd'] = (_do_.square().sum(1) - 1).square().mean()
            
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

        if params['model_type'] == 'sqJ_orth_cert':
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
