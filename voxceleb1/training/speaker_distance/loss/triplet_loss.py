import torch

from . import dist
from . import pair


def batch_hard(X, margin=1.0):
    """Implements the batch-hard triplet loss.

    See <https://arxiv.org/pdf/1703.07737.pdf> for the original proposal.

    Parameters :
    X : torch Tensor of shape (p, k, d), where p is the number of unique
        subjects, k is the number of unique samples per subject, and
        d is the number of features per sample.
    """
    p, k, d = X.size()
    pairwise_distances = dist.pairwise(X)
    inside, outside = pair.inside_out(pairwise_distances, p, k)
    value = margin + inside.max(dim=-1)[0] - outside.min(dim=-1)[0]
    return torch.nn.functional.relu(value).mean()
