### Inspired from https://github.com/cemanil/LNets/blob/master/lnets/models/activations/group_sort.py on Tuesday, December 15th 2020

import numpy as np
import torch

class GroupSort(torch.nn.Module):

    def __init__(self, num_units, axis=-1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis
        self.indices = None

    def forward(self, x):
        n, m = x.shape
        group_sorted, indices = group_sort(x, self.num_units, self.axis)
        assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "
        self.indices = indices
        self.indices_t = get_transpose_indices(indices)
        return group_sorted
    
    def apply_jacobian(self, x):
        return torch.gather(x, 1, self.indices_t)

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def process_group_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size


def group_sort(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, indices = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))
    indices = indices.view(*list(x.shape))
    return sorted_x, indices

def get_permutation_matrix(indices):
    n, m = indices.shape
    return torch.zeros((n,m,m)).scatter(2, indices.unsqueeze(2), 1)

def get_transpose_indices(indices):
    n, m = indices.shape
    return indices.scatter(1, indices,  torch.arange(m).expand(n, m))

def check_group_sorted(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)

    x_np = x.cpu().data.numpy()
    x_np = x_np.reshape(*size)
    axis = axis if axis == -1 else axis + 1
    x_np_diff = np.diff(x_np, axis=axis)

    # Return 1 iff all elements are increasing.
    if np.sum(x_np_diff < 0) > 0:
        return 0
    else:
        return 1
