def euclidean_dist(V1, V2):
    return (V1 - V2).norm(dim=-1)


def pairwise(X, dist_fn=euclidean_dist):
    """Compute the specified distance metric between X1 and X2.

    Parameters :
    X : torch Tensor of size (N, *, D), where N is the batch size
        and D is the length of the feature vector.
    dist_fn : function that takes in two torch.Tensor and outputs
        the torch Tensor distance between them.

    Output :
    pairwise_dist : torch Tensor of size (N*, N*), pairwise distance
        between each element of X and every other element.
    """
    D = X.size(-1)
    V1 = X.view(-1, 1, D)
    V2 = X.view(1, -1, D)
    return dist_fn(V1, V2)
