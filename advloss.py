from __future__ import absolute_import
import torch
import random
import numpy as np
from torch import nn

__all__ = ['DeepSupervision', 'adv_CrossEntropyLoss','adv_CrossEntropyLabelSmooth', 'adv_TripletLoss']

def DeepSupervision(criterion, xs, *args, **kwargs):
  loss = 0.
  for x in xs: loss += criterion(x, *args, **kwargs)
  return loss

class adv_CrossEntropyLoss(nn.Module):
  def __init__(self, use_gpu=True):
    super(adv_CrossEntropyLoss, self).__init__()
    self.use_gpu = use_gpu
    self.crossentropy_loss = nn.CrossEntropyLoss()

  def forward(self, logits, pids):
    """
    Args:
        logits: prediction matrix (before softmax) with shape (batch_size, num_classes)
    """
    _, adv_target = torch.min(logits, 1)

    if self.use_gpu: adv_target = adv_target.cuda()
    loss = self.crossentropy_loss(logits, adv_target)
    return torch.log(loss)

class adv_CrossEntropyLabelSmooth(nn.Module):
  """
  Args:
      num_classes (int): number of classes.
      epsilon (float): weight.
  """
  def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
    super(adv_CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.use_gpu = use_gpu
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, logits, pids):
    """
    Args:
        logits: prediction matrix (before softmax) with shape (batch_size, num_classes)
        pids: ground truth labels with shape (num_classes)
    """
    # n = pids.size(0)
    # _, top2 = torch.topk(logits, k=2, dim=1, largest=False)
    # adv_target = top2[:,0]
    # for i in range(n):
    #   if adv_target[i] == pids[i]: adv_target[i] = top2[i,1]
    #   else: continue
    _, adv_target = torch.min(logits, 1)
    # for i in range(n):
    #   while adv_target[i] == pids[i]:
    #     adv_target[i] = random.randint(0, self.num_classes)

    log_probs = self.logsoftmax(logits)
    adv_target = torch.zeros(log_probs.size()).scatter_(1, adv_target.unsqueeze(1).data.cpu(), 1)
    smooth = torch.ones(log_probs.size()) / (self.num_classes-1)
    smooth[:, pids.data.cpu()] = 0 # Pytorch1.0
    smooth = smooth.cuda()
    if self.use_gpu: adv_target = adv_target.cuda()
    adv_target = (1 - self.epsilon) * adv_target + self.epsilon * smooth
    loss = (- adv_target * log_probs).mean(0).sum()
    return torch.log(loss)

class adv_TripletLoss(nn.Module):
  def __init__(self, ak_type, margin=0.3):
    super(adv_TripletLoss, self).__init__()
    self.margin = margin
    self.ak_type = ak_type
    self.ranking_loss = nn.MarginRankingLoss(margin=margin)

  def forward(self, features, pids, targets=None):
      """
      Args:
          features: feature matrix with shape (batch_size, feat_dim)
          pids: ground truth labels with shape (num_classes)
          targets: pids with certain attribute (batch_size, pids)
      """
      n = features.size(0)

      dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
      dist = dist + dist.t()
      dist.addmm_(1, -2, features, features.t())
      dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

      if self.ak_type < 0: 
        mask = pids.expand(n, n).eq(pids.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
          dist_an.append(dist[i][mask[i]].min().unsqueeze(0)) # make nearest pos-pos far away
          dist_ap.append(dist[i][mask[i] == 0].max().unsqueeze(0)) # make hardest pos-neg closer

      elif self.ak_type > 0: 
        p = []
        for i in range(n):
          p.append(pids[i].item())
        mask = targets[0][p].expand(n, n).eq(targets[0][p].expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
          dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
          dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

      dist_ap = torch.cat(dist_ap)
      dist_an = torch.cat(dist_an)

      y = torch.ones_like(dist_an)

      loss = self.ranking_loss(dist_an, dist_ap, y)
      return torch.log(loss)