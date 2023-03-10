import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model.layer import Conv2d
import numpy as np


class BasicBlock(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    '''Bottleneck.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class RPshortcut(nn.Module):
    '''Shortcut with random projection.'''
    def __init__(self, in_planes, planes, stride=1, rp_out_channel=0, rp_feature_size=0):
        super(RPshortcut, self).__init__()
        self.shortcut_rp_conv = Conv2d(in_planes, planes - rp_out_channel, kernel_size=1, stride=stride, bias=False)
        self.shortcut_rp = nn.Conv2d(in_planes, rp_out_channel, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(planes)

        self.rp_feature_size = rp_feature_size

        # disable grad for rp
        self.shortcut_rp.weight.requires_grad = False

    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        out = torch.cat([out, rp_out], dim=1)
        return out

    def forward(self, x):

        out = self.shortcut_rp_conv(x)
        out = self.rp_forward(x, out, self.shortcut_rp)
        out = self.shortcut_bn(out)

        return out


class BasicRPBlock(nn.Module):
    '''BasicBlock with random projection.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, rp_out_channel=0):
        super(BasicRPBlock, self).__init__()
        self.rp_conv1 = Conv2d(in_planes, planes - rp_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.rp1 = nn.Conv2d(in_planes, rp_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.rp1.weight.requires_grad = False
        self.bn1 = nn.BatchNorm2d(planes)

        self.rp_conv2 = Conv2d(planes, planes - rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.rp2 = nn.Conv2d(planes, rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.rp2.weight.requires_grad = False
        self.bn2 = nn.BatchNorm2d(planes)

        self.in_planes = in_planes
        self.planes = planes
        self.rp_out_channel = rp_out_channel

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        out = torch.cat([out, rp_out], dim=1)
        return out

    def forward(self, x):

        out = self.rp_conv1(x)
        out = self.rp_forward(x, out, self.rp1)
        out = F.relu(self.bn1(out))

        conv_out = self.rp_conv2(out)
        out = self.rp_forward(out, conv_out, self.rp2)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class RPBottleneck(nn.Module):
    '''Bottleneck.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, rp_out_channel=0):
        super(RPBottleneck, self).__init__()
        self.rp_conv1 = Conv2d(in_planes, planes - rp_out_channel, kernel_size=1, bias=False)
        self.rp1 = nn.Conv2d(in_planes, rp_out_channel, kernel_size=1, bias=False)
        self.rp1.weight.requires_grad = False
        self.bn1 = nn.BatchNorm2d(planes)

        self.rp_conv2 = Conv2d(planes, planes - rp_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.rp2 = nn.Conv2d(planes, rp_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.rp2.weight.requires_grad = False
        self.bn2 = nn.BatchNorm2d(planes)

        self.rp_conv3 = Conv2d(planes, self.expansion*planes - rp_out_channel, kernel_size=1, bias=False)
        self.rp3 = nn.Conv2d(planes, rp_out_channel, kernel_size=1, bias=False)
        self.rp3.weight.requires_grad = False
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.in_planes = in_planes
        self.planes = planes
        self.rp_out_channel = rp_out_channel

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        out = torch.cat([out, rp_out], dim=1)
        return out

    def forward(self, x):
        out = self.rp_conv1(x)
        out = self.rp_forward(x, out, self.rp1)
        out = F.relu(self.bn1(out))

        conv_out = self.rp_conv2(out)
        out = self.rp_forward(out, conv_out, self.rp2)
        out = F.relu(self.bn2(out))

        conv_out = self.rp_conv3(out)
        out = self.rp_forward(out, conv_out, self.rp3)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, rpblock, num_blocks, num_classes=10, rp=False, rp_block=None, rp_out_channel=0,
                 normalize=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.rp = rp
        self.rp_block = rp_block

        if rp and -1 in rp_block:
            self.rp_conv1 = Conv2d(3, 64 - rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.rp1 = nn.Conv2d(3, rp_out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.rp1.weight.requires_grad = False
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, rpblock, 64, num_blocks, stride=1, block_id=0, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)
        self.layer2 = self._make_layer(block, rpblock, 128, num_blocks, stride=2, block_id=1, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)
        self.layer3 = self._make_layer(block, rpblock, 256, num_blocks, stride=2, block_id=2, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)
        self.layer4 = self._make_layer(block, rpblock, 512, num_blocks, stride=2, block_id=3, rp=rp,
                                       rp_block=rp_block, rp_out_channel=rp_out_channel)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.normalize = normalize

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def random_rp_matrix(self):
        for name, param in self.named_parameters():
            if 'rp' in name and 'conv' not in name:
                kernel_size = param.data.size()[-1]
                param.data = torch.normal(mean=0.0, std=1/kernel_size, size=param.data.size()).to('cuda')

    def _make_layer(self, block, rpblock, planes, num_blocks, stride, block_id, rp, rp_block, rp_out_channel):
        num_block = num_blocks[block_id]
        strides = [stride] + [1]*(num_block-1)
        layers = []
        # get the indices of blocks for rp
        if rp:
            rp_blocks = np.arange(rp_block[0], rp_block[1]+1)
            block_id_sum = 0
            for i in range(0, block_id):
                block_id_sum += num_blocks[i]
        for stride in strides:
            if rp and block_id_sum in rp_blocks:
                layers.append(rpblock(self.in_planes, planes, stride, rp_out_channel))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            block_id_sum += 1
        return nn.ModuleList(layers)

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
        out = F.relu(self.bn1(out))

        for i, op in enumerate(self.layer1):
            out = op(out)
        for i, op in enumerate(self.layer2):
            out = op(out)
        for i, op in enumerate(self.layer3):
            out = op(out)
        for i, op in enumerate(self.layer4):
            out = op(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet18(num_classes=10, rp=False, rp_block=None, rp_out_channel=0, normalize=None):
    return ResNet(BasicBlock, BasicRPBlock, [2,2,2,2], num_classes=num_classes, rp=rp, rp_block=rp_block,
                  rp_out_channel=rp_out_channel, normalize=normalize)

def ResNet50(num_classes=10, rp=False, rp_block=None, rp_out_channel=0, normalize=None):
    return ResNet(Bottleneck, RPBottleneck, [3,4,6,3], num_classes=num_classes, rp=rp, rp_block=rp_block,
                  rp_out_channel=rp_out_channel, normalize=normalize)