import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.kernel_size_int = kernel_size

    def forward(self, x):
        weight = self.weight
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        weight = weight / self.kernel_size_int
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)