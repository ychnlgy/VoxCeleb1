import torch

class Parallel(torch.nn.Module):

    def __init__(self, br1, br2):
        super().__init__()
        self.br1 = br1
        self.br2 = br2

    def forward(self, X):
        h1 = self.br1(X)
        h2 = self.br2(X)
        return 0.5 * (h1 + h2)
