from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['ResNet50']

class ResNet50(nn.Module):
  """
  Alignedreid: Surpassing human-level performance in person re-identification

  Reference:
  Zhang, Xuan, et al. "Alignedreid: Surpassing human-level performance in person re-identification." arXiv preprint arXiv:1711.08184 (2017)
  """
  def __init__(self, num_classes, **kwargs):
    super(ResNet50, self).__init__()
    self.loss = {'softmax', 'metric'}
    resnet50 = torchvision.models.resnet50(pretrained=True)
    self.base = nn.Sequential(*list(resnet50.children())[:-2])
    self.classifier = nn.Linear(2048, num_classes)
    self.feat_dim = 2048 # feature dimension
    self.aligned = True
    self.horizon_pool = HorizontalMaxPool2d()
    if self.aligned:
      self.bn = nn.BatchNorm2d(2048)
      self.relu = nn.ReLU(inplace=True)
      self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

  def forward(self, x, is_training):
    x = self.base(x)
    if not is_training:
      lf = self.horizon_pool(x)
    if self.aligned and is_training:
      lf = self.bn(x)
      lf = self.relu(lf)
      lf = self.horizon_pool(lf)
      lf = self.conv1(lf)
    if self.aligned or not is_training:
      lf = lf.view(lf.size()[0:3])
      lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
    x = F.avg_pool2d(x, x.size()[2:])
    f = x.view(x.size(0), -1)
    #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
    if not is_training:
      return [f,lf]
    y = self.classifier(f)
    if self.loss == {'softmax'}:
      return [y]
    elif self.loss == {'metric'}:
      if self.aligned: 
        return [f, lf]
      return [f]
    elif self.loss == {'softmax', 'metric'}:
      if self.aligned: 
        return [y, f, lf]
      return [y, f]
    else:
      raise KeyError("Unsupported loss: {}".format(self.loss))

class HorizontalMaxPool2d(nn.Module):
  def __init__(self):
    super(HorizontalMaxPool2d, self).__init__()


  def forward(self, x):
    inp_size = x.size()
    return nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))