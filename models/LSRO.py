from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

__all__ = ['DenseNet121']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121,self).__init__()
        model_ft = models.densenet121(pretrained=True)
        # add pooling to the model
        # in the originial version, pooling is written in the forward function 
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(1024, num_bottleneck)]  #For ResNet, it is 2048
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, num_classes)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x, is_training):
        x = self.model.features(x)  
        x = x.view(x.size(0),-1)
        x = self.model.fc(x)
        logits = self.classifier(x)
        return [logits, x]

