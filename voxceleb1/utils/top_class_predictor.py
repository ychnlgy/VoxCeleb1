import torch


class TopClassPredictor:

    def __init__(self, top_ks):
        self.acc = [0] * len(top_ks)
        self.n = 0
        self.top_ks = top_ks

    def update(self, yh, y):
        """Stores accuracy values from the predicted classes.

        Parameters :
        yh : torch FloatTensor of size (N, C), where N is the
            size of the batch and C is the number of classes.
        y : torch LongTensor of size (N), the class labels.
        """
        with torch.no_grad():
            _, choices = yh.sort()
            n = len(yh)
            matches = (choices == y.unsqueeze(1)).float()
            for i, top_k in enumerate(self.top_ks):
                count = matches[:, :top_k].sum(dim=1)
                acc = count.mean().item()
                self.acc[i] = self._average(self.acc[i], acc, n)
            self.n += n

    def peek(self):
        return self.acc

    # === PRIVATE ===

    def _average(self, curr_acc, acc, n):
        new_n = n + self.n
        p1 = n / new_n * acc
        p2 = self.n / new_n * curr_acc
        return p1 + p2
