import torch


def inside_out(dist, p, k):
    """Return same and different subject distances.

    Parameters :
    dist : torch Tensor of shape (p*k, p*k), where p and k
        are described as follows. Represents the pairwise
        distances of a (p, k, D) torch Tensor, where D is
        the number of features.
    p : int number of subjects.
    k : int number of samples per subject.

    Output :
    inside : torch Tensor of shape (p * k, k), represents the
        distances of samples to other samples of the same subject.
    outside : torch Tensor of shape (p * k, (p - 1) * k), represents
        the distances of samples to each sample of all different subjects.
    """
    index = _get_indices(dist, p, k)
    inside = dist[index].view(p*k, k)
    outside = dist[~index].view(p*k, (p-1)*k)
    return inside, outside


# === PRIVATE ===


def _get_indices(dist, p, k):
    w, h = dist.size()
    assert w == h == p * k
    index = torch.eye(p, dtype=torch.uint8)
    index = index.view(p, 1, p, 1).repeat(1, k, 1, k)
    return index.view(p*k, p*k)
