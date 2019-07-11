import sklearn.metrics

def equal_error(y, yh):
    """Computes the score and best threshold for equal error rates.

    Copied from :
    https://stackoverflow.com/questions/28339746/equal-error-rate-in-python

    Description :
    We wish to compute the threshold for achieving equal false positive
    rates to false negative rates.

    Parameters :
    y : numpy bool array of shape (N), where N is the number of targets.
        We assume that 0 means negative and 1 means positive.
    yh : numpy float array of shape (N). We wish to determine the threshold
        of these values that achieves the best equal error rate with y.

    Output :
    score : equal error rate score (i.e. false positive rate at the optimal
        threshold in respect to equal error rates).
    threshold : the threshold to achieve the score.
    """
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y, yh)
    fnr = 1 - tpr
    index = numpy.argmin(numpy.abs(fnr - fpr))
    return fpr[index], threshold[index]
    
