import multiprocessing

import tqdm


class WorkerManager:

    def __init__(self, num_workers):
        self.num_workers = num_workers

    def imap(self, f, args):
        with multiprocessing.Pool(self.num_workers) as pool:

            # Run as many cores as possible
            async_placeholders = [
                pool.apply_async(f, (arg,)) for arg in args
            ]

            # Iterate results sequentially
            for placeholder in tqdm.tqdm(
                async_placeholders,
                ncols=80,
                desc="Multi (%d) processing" % self.num_workers
            ):
                placeholder.wait()
                yield placeholder.get()


if __name__ == "__main__":

    import time

    import numpy


    def f(x):
        time.sleep(1)
        return x.mean().item()


    manager = WorkerManager(num_workers=3)

    args = [
        numpy.array([0, 1, 2]),
        numpy.array([1, 2, 3]),
        numpy.array([2, 3, 4]),
        numpy.array([3, 4, 5]),
        numpy.array([4, 5, 6]),
        numpy.array([5, 6, 7]),
        numpy.array([6, 7, 8])
    ]

    t0 = time.time()
    results = []
    for result in manager.imap(f, args):
        results.append(result)

    assert abs(
        numpy.array(results) - numpy.arange(1, 8)
    ).astype(float).sum() < 1e-8

    assert time.time() - t0 < 4.0, "One worker takes 7 seconds"
