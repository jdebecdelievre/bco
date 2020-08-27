#!/usr/bin/env python3
import torch

from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from gpytorch.kernels.matern_kernel import MaternKernel, MaternCovariance
from gpytorch.kernels import Kernel
from math import sqrt

class Matern52KernelGrad(MaternKernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super(Matern52KernelGrad, self).__init__(nu=5/2, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        if diag:
            raise NotImplementedError

        K = torch.zeros(*batch_shape, n1 * (d + 1), n2 * (d + 1), device=x1.device, dtype=x1.dtype)
        # Scale input and Ouputs for stability
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        distance = sqrt(5) * self.covar_dist(x1_, x2_, **params)
        sqdistance = distance.square()

        # Form all possible rank-1 products for the gradient and Hessian blocks
        d_dist =  5 * ((x1_.view(*batch_shape, n1, 1, d) - x2_.view(*batch_shape, 1, n2, d)) / distance.unsqueeze(-1))
        d_dist = d_dist / self.lengthscale.unsqueeze(-2)
        d_dist = torch.transpose(d_dist, -1, -2).contiguous()

        # 1) Kernel Block
        exp_component = torch.exp(-distance)
        constant_component = 1 + distance + sqdistance/3
        K[..., :n1, :n2] = constant_component * exp_component

        # 2) First Gardient Block
        grad_component = (sqdistance + distance)/3
        d_dist1 = d_dist.view(*batch_shape, n1, n2 * d)
        K[..., :n1, n2:] = d_dist1 * (grad_component * exp_component).repeat([*([1] * (n_batch_dims + 1)), d])

        # 3) Second gradient block
        d_dist2 = d_dist.transpose(-1, -3).reshape(*batch_shape, n2, n1 * d)
        d_dist2 = d_dist2.transpose(-1, -2)
        K[..., n1:, :n2] = -d_dist2 * (grad_component * exp_component).repeat([*([1] * n_batch_dims), d, 1])

        # 4) Hessian block
        hess_component = (sqdistance - distance - 1)/3
        d_dist3 = d_dist1.repeat([*([1] * n_batch_dims), d, 1]) * d_dist2.repeat([*([1] * (n_batch_dims + 1)), d])

        kp = KroneckerProductLazyTensor(
            (torch.eye(d, d, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1)) * 5 / self.lengthscale.pow(2),
            torch.ones(n1, n2, device=x1.device, dtype=x1.dtype).repeat(*batch_shape, 1, 1) / distance,
        )

        d2_dist = (kp.evaluate() - d_dist3 / distance.repeat([*([1] * n_batch_dims), d, d]))
        chain_rule = d2_dist * grad_component.repeat([*([1] * n_batch_dims), d, d]) - \
                        d_dist3 * hess_component.repeat([*([1] * n_batch_dims), d, d])
        K[..., n1:, n2:] = chain_rule * exp_component.repeat([*([1] * n_batch_dims), d, d])

        # Symmetrize for stability
        if n1 == n2 and torch.eq(x1, x2).all():
            K = 0.5 * (K.transpose(-1, -2) + K)

        # Apply a perfect shuffle permutation to match the MutiTask ordering
        pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
        pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
        K = K[..., pi1, :][..., :, pi2]

        return K
        

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1) + 1
