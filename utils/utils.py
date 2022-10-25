import torch
import math
from torch.autograd import Function

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

def ramp_up(x, lamparam=0.1):
    r""" Adaptive loss weight scheduler
         lamparam(float): weight increase damping ratio
         """
    if x > 1.0:
        return 1.0
    return sigmoid_ramp_up(x, lamparam)

def sigmoid_ramp_up(x, lamparam):
    den = 1.0 + math.exp(-x/lamparam) # for low increase ratio
    lamb = 2.0 / den - 1.0
    return lamb

class L2dist(Function):
    def __init__(self, p):
        super(L2dist, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)