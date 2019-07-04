import random

import numpy
import torch


class SeedBubble:

    def __init__(self, seed):
        self.seed = seed
        self.numpy_state = None
        self.torch_state = None

    def __enter__(self):
        """Return random.Random instance."""
        self.numpy_state = numpy.random.get_state()
        self.torch_state = torch.get_rng_state()
        return random.Random(self.seed)

    def __exit__(self, *args):
        numpy.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
