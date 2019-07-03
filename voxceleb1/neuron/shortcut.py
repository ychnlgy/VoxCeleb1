"""A trick shortcut for large improvement to residual neural networks.

Copied from
https://github.com/owruby/shake-shake_pytorch/blob/master/models/shakeshake.py
"""

import torch


class Shortcut(torch.nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        assert out_ch % 2 == 0
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv1 = self._create_conv()
        self.conv2 = self._create_conv()
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = torch.nn.functional.relu(x)

        h1 = torch.nn.functional.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(h, (-1, 1, -1, 1)), 1, self.stride
        )
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)

    def _create_conv(self):
        return torch.nn.Conv2d(
            self.in_ch,
            self.out_ch//2,
            1,
            stride=1,
            padding=0,
            bias=False
        )
