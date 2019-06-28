"""Functions that are commonly used on tensors."""


import numpy
import torch
import torch.utils.data


def create_loader(tensors, **kwargs):
    """Return the DataLoader for the input list of tensors."""
    dataset = torch.utils.data.TensorDataset(*tensors)
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader


def one_hot(I, classes):
    """Produce the one-hot encoding of the index vector I.
    Parameters :
    I : torch LongTensor of size (N). Indices for onehot encoding.
    classes : int number of columns in the onehot encoding.
    Output:
    onehot - torch FloatTensor of size (N, classes).
    """
    out = torch.zeros(I.size(0), classes).to(I.device)
    return out.scatter_(1, I.unsqueeze(1), 1)


def param_count(m):
    """Return the int number of trainable parameters in module m."""
    return sum(torch.numel(p) for p in m.parameters() if p.requires_grad)


def rand_indices(n):
    """Return shuffled indices in a torch.LongTensor array of length n."""
    indices = torch.arange(n).long()
    numpy.random.shuffle(indices.numpy())
    return indices
