import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from model.layer import Conv2d


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class BasicRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, rp_out_channel=0):
        super(BasicRPBlock, self).__init__()

        self.rp_out_channel = rp_out_channel

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.rp_conv1 = Conv2d(in_planes, out_planes - rp_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.rp1 = nn.Conv2d(in_planes, rp_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.rp1.weight.requires_grad = False

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.rp_conv2 = Conv2d(out_planes, out_planes - rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.rp2 = nn.Conv2d(out_planes, rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.rp2.weight.requires_grad = False
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        out = torch.cat([out, rp_out], dim=1)
        return out

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        conv_out = self.rp_conv1(out if self.equalInOut else x)
        out = self.rp_forward(out if self.equalInOut else x, conv_out, self.rp1)

        out = self.relu2(self.bn2(out))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        conv_out = self.rp_conv2(out)
        out = self.rp_forward(out, conv_out, self.rp2)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, block_id=0, rp=False,
                 rp_block=None, rp_out_channel=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, block_id, rp,
                                      rp_block, rp_out_channel)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, block_id, rp, rp_block,
                    rp_out_channel):
        layers = []
        # get the indices of blocks for rp
        rp_blocks = np.arange(rp_block[0], rp_block[1]+1)
        block_id = block_id*5
        for i in range(int(nb_layers)):
            if rp and block_id in rp_blocks:
                layers.append(BasicRPBlock(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                           dropRate, rp_out_channel))
            else:
                layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
            block_id += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, rp=False, rp_block=None,
                 rp_out_channel=0, normalize=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        # random projection
        self.rp = rp
        self.rp_block = rp_block

        # 1st conv before any network block
        if rp and -1 in rp_block:
            self.rp_conv1 = Conv2d(3, nChannels[0] - rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.rp1 = nn.Conv2d(3, rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.rp1.weight.requires_grad = False
        else:
            self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, block_id=0, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, block_id=1, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, block_id=2, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)

        self.blocks = [self.block1, self.block2, self.block3]

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        if out is None:
            return rp_out
        else:
            out = torch.cat([out, rp_out], dim=1)
            return out

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        if self.rp and -1 in self.rp_block:
            out = self.rp_conv1(x)
            out = self.rp_forward(x, out, self.rp1)
        else:
            out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        output = self.fc(out)

        return output

    def random_rp_matrix(self):
        for name, param in self.named_parameters():
            if 'rp' in name and 'conv' not in name:
                kernel_size = param.data.size()[-1]
                param.data = torch.normal(mean=0.0, std=1/kernel_size, size=param.data.size()).to('cuda')


def WideResNet32(num_classes=10, rp=False, rp_block=None, rp_out_channel=0, normalize=None):
    return WideResNet(num_classes=num_classes, rp=rp, rp_block=rp_block, rp_out_channel=rp_out_channel,
                      normalize=normalize)