"""Implementation of residual connection."""

import torch


ID = torch.nn.Sequential()


class ResBlock(torch.nn.Module):
    """Residual connection."""

    def __init__(self, block, weight=1, shortcut=ID, activation=ID):
        """Instantiate a residual block connection.

        Parameters :
        block : torch.nn.Module branch of non-linear transformation.
        weight : int balance factor between the shortcut and the branch.
        shortcut : torch.nn.Module transformation of the shortcut.
        activation : torch.nn.Module transformation of the overall output.

        Note :
        The parameter "weight" balances the outputs. Typically, we use 1
        to sum the shortcut with the branch transformation. Other times,
        we average them with weight=0.5 maintain the magnitude of inputs, .
        """
        super(ResBlock, self).__init__()
        self.block = block
        self.shortcut = shortcut
        self.activation = activation
        self.weight = weight

    def forward(self, X):
        """Return the residual transformation of tensor X."""
        out = (self.block(X) + self.shortcut(X)) * self.weight
        return self.activation(out)
