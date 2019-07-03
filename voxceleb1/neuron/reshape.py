"""Module for reshaping tensors."""

import torch


class Reshape(torch.nn.Module):
    """Reshape batched input tensor."""

    def __init__(self, *size):
        """Input size per batch element."""
        super().__init__()
        self.size = size

    def forward(self, X):
        """Resizes tensor X of initial size (N, *)."""
        return X.view(len(X), *self.size)
