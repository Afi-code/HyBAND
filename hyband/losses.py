import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .models.bands import radius_map, raised_cosine

class MixedL1L2Loss(nn.Module):

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        b = pred.shape[0]
        d = (pred - target).reshape(b, -1)
        t = target.reshape(b, -1)
        l2 = torch.linalg.vector_norm(d, ord=2, dim=1) / \
            (torch.linalg.vector_norm(t, ord=2, dim=1) + self.eps)
        l1 = torch.linalg.vector_norm(d, ord=1, dim=1) / \
            (torch.linalg.vector_norm(t, ord=1, dim=1) + self.eps)
        return (0.5 * l2 + 0.5 * l1).mean()
