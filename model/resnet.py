from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torchvision import models
from model.basenet import PredictorWN_deep

class ResNet(nn.Module):
    def __init__(self, args, net, pretrained=True):
        super(ResNet, self).__init__()

        ### backbone selection
        if net == 'resnet50':
            model_resnet = models.resnet50(pretrained=pretrained)
            inc = 2048
        else:
            raise NotImplementedError

        ### rename resnet blocks
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        ### Define classifier
        self.classifier = PredictorWN_deep(num_class=args.ncls, inc=inc, num_emb=512, temp=0.1)

        ### misc
        self.frozen_layer_list = []

    def forward(self, x, reverse=False, getemb=False, getfeat=False, normemb=True):
        assert(not (getemb and getfeat))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        feat = x.view(x.size(0), -1)
        out = self.classifier(feat, reverse=reverse, getemb=getemb, normemb=normemb)
        out = (out, feat) if getfeat else out
        
        return out

    def get_intermediate_feat(self, x, group_list=[0,1,2,3,4], getemb=True, normemb=True):
        c1 = self.conv1(x)
        x = self.bn1(c1)
        x = self.relu(x)
        x = self.maxpool(x)
        g1 = self.layer1(x)
        g2 = self.layer2(g1)
        g3 = self.layer3(g2)
        g4 = self.layer4(g3)
        x = self.avgpool(g4)
        
        feat = x.view(x.size(0), -1)
        out = self.classifier(feat, reverse=False, getemb=getemb, normemb=normemb)

        feat_list = [c1, g1, g2, g3, g4]
        out_feat_list = [feat_list[i] for i in group_list]

        return out_feat_list, out # list, pred, embedding

    def trainable_parameters(self):
        backbone_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        backbone_params = []
        for layer in backbone_layers:
            backbone_params += [param for param in layer.parameters()]
        classifier_params = list(self.classifier.parameters())

        return backbone_params, classifier_params

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