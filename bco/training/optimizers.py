from torch.optim import Adam, SGD
from bco.training.alig import AliG
from lnets.optimizers.aggmo import AggMo
import torch.optim.lr_scheduler as lr_scheduler
from bco.training.adahessian import Adahessian

def get_optimizer(optim_params, model):
    # Optimizer
    if optim_params['optimizer'].lower() == 'adam':
        opt = Adam(model.parameters(),
                            lr=optim_params['step_size'],
                            weight_decay=optim_params['weight_decay'],
                            betas=(optim_params['momentum'], 0.999), 
                            eps=1e-08
        )

    elif optim_params['optimizer'].lower() == 'sgd':
        opt = SGD(model.parameters(),
                            lr=optim_params['step_size'],
                            weight_decay=optim_params['weight_decay']
        )

    elif optim_params['optimizer'].lower() == 'alig':
        opt = AliG(model.parameters(), max_lr=optim_params['step_size'])

    elif optim_params['optimizer'].lower() == 'aggmo':
        opt = AggMo(model.parameters(), lr=optim_params['step_size'], momentum=optim_params['betas'])
    
    elif optim_params['optimizer'].lower() == 'adahessian':
        opt = Adahessian(model.parameters(),                             lr=optim_params['step_size'],
                            weight_decay=optim_params['weight_decay'],
                            betas=(optim_params['momentum'], 0.999), 
                            eps=1e-08)
    # Scheduler
    schedule_params = optim_params['lr_scheduler']
    if optim_params['optimizer'].lower() == 'alig':
        return opt, None
    if schedule_params['name'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                        patience=schedule_params['patience'], 
                        factor=schedule_params['factor'])
    elif schedule_params['name'] == 'exp':
        scheduler = lr_scheduler.ExponentialLR(opt, 
                schedule_params['lr_decay'], schedule_params['last_epoch'])
    elif schedule_params['name'] == 'step':
        scheduler = lr_scheduler.MultiStepLR(opt, 
                        schedule_params['milestones'], schedule_params['lr_decay'])
    
    return opt, scheduler
