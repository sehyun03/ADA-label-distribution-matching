from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Function
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from utils.layer_utils import load_weights
from copy import deepcopy
import functools
__all__ = ['ResNet', 'resnet34', 'resnet50']

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101':
        'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152':
        'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nobn = nobn

    def forward(self, x, source=True):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.nobn = nobn

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, pretrained, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.clayers = [64, 128, 256, 512]
        self.frozen_layer_list = []

        if(block == 'basic'):
            block = BasicBlock
        elif(block == 'bottleneck'):
            block = Bottleneck
        else:
            assert(isinstance(block, nn.Module))

        self.layer1 = self._make_layer(block, self.clayers[0], layers[0])
        self.layer2 = self._make_layer(block, self.clayers[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.clayers[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.clayers[3], layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if(pretrained):
            ### initialize backbone with imagenet pretrained model
            pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
            load_weights(self, pretrained_dict, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1, nobn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nobn=nobn))
        return nn.Sequential(*layers)

    def freeze_layers(self, layer_list):
        for layer in layer_list:
            ### kill gradient
            for param in layer.parameters():
                param.requires_grad = False

            ### evaluation mode for batchnorm
            for layer in layer_list:
                for m in layer.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        self.frozen_layer_list = layer_list

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        ### eval mode on frozen layers
        for layer in self.frozen_layer_list:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        return self
    
    def set_bn_mode(self, mode='train'):
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                if mode == 'train':
                    m.train()
                elif mode == 'eval':
                    m.eval()
                else:
                    raise NotImplementedError

    def set_bn_requires_grad(self, requires_grad=True):
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = requires_grad
                m.bias.requires_grad = requires_grad

    def forward(self, x):
        raise NotImplementedError