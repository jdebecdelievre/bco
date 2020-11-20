# Copied from https://github.com/sprillo/softsort/blob/master/pytorch/neuralsort_cpu_or_gpu.py
# and https://github.com/sprillo/softsort/blob/master/pytorch/softsort.py 
# on October 18th 2020.
# Reference: SoftSort: A Continuous Relaxation for the argsort Operator
# Prillo and Eisenschlos, 2020. https://arxiv.org/pdf/2006.16038.pdf

import torch
from torch import Tensor
torch.set_default_dtype(torch.float64)


class SoftSort(torch.nn.Module):
    def __init__(self, tau=1e-6, hard=False, pow=2.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        # self.tau = torch.nn.Parameter(torch.tensor(tau))
        self.pow = pow

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return (P_hat @ scores).squeeze() 


class SoftSort_p1(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(SoftSort_p1, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat @ scores


class SoftSort_p2(torch.nn.Module):
    def __init__(self, tau=.0001, hard=False):
        super(SoftSort_p2, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = ((scores.transpose(1, 2) - sorted) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat @ scores

class NeuralSort(torch.nn.Module):
    def __init__(self, tau=1e-6, hard=False, device='cpu'):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self._tau = tau
        # self.tau = torch.nn.Parameter(torch.tensor(1.))
        self.tau = 1.
        self.device = device
        if device == 'cuda':
            self.torch = torch.cuda
        elif device == 'cpu':
            self.torch = torch
        else:
            raise ValueError('Unknown device: %s' % device)

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        tau = self._tau * self.tau

        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = self.torch.DoubleTensor(dim, 1).fill_(1)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        # B = torch.matmul(A_scores, torch.matmul(
        #     one, torch.transpose(one, 0, 1)))  # => NeuralSort O(n^3) BUG!
        B = torch.matmul(torch.matmul(A_scores,
                         one), torch.transpose(one, 0, 1))  # => Bugfix
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(self.torch.DoubleTensor)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=self.device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(self.torch.LongTensor)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(self.torch.LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        
        return (P_hat @ scores).squeeze()