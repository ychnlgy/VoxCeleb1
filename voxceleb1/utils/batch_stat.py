import numpy


class BatchStat:

    def __init__(self, axis):
        self.miu = 0
        self.var = 0
        self.n = 0
        self.axis = axis

    def update(self, x):
        """Update statistics given the batch item.

        Parameters :
        x : numpy array with ndim > self.axis.
        """
        n = x.shape[self.axis]
        miu = x.mean(axis=self.axis)
        new_n = self.n + n
        self.var = self._calc_var(x, miu, n, new_n)
        self.miu = self._calc_miu(miu, n, new_n)
        self.n = new_n

    def peek(self):
        """Return the mean and variance of all encountered batches."""
        miu = numpy.expand_dims(self.miu, self.axis)
        std = numpy.expand_dims(numpy.sqrt(self.var), self.axis)
        return miu, std

    # === PRIVATE ===

    def _calc_var(self, x, miu, n, new_n):
        var = x.var(axis=self.axis)
        pvar1 = n / new_n * var
        pvar2 = self.n / new_n * self.var
        fmean = n * self.n / (n + self.n)**2 * (miu - self.miu)**2
        return pvar1 + pvar2 + fmean
        
    def _calc_miu(self, miu, n, new_n):
        return n / new_n * miu + self.n / new_n * self.miu


if __name__ == "__main__":
    stat = BatchStat(-1)

    X = numpy.zeros((1000, 4, 100))

    for i in range(1000):
        x = numpy.random.randn(4, 100)
        stat.update(x)
        X[i] = x

    X = numpy.transpose(X, [1, 0, 2]).reshape(4, -1)

    miu1, std1 = stat.peek()
    miu2, std2 = (
        X.mean(axis=-1).reshape(-1, 1),
        X.std(axis=-1).reshape(-1, 1)
    )

    eps = 1e-8
    assert abs(miu1 - miu2).sum().item() < eps
    assert abs(std1 - std2).sum().item() < eps
