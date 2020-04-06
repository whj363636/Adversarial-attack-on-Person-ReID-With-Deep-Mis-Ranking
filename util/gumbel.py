import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


def _sample_gumbel(shape, eps=1e-10, out=None):
  """
  Based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
  (MIT license)
  """
  U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
  return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, T=1, eps=1e-10):
  """
  Based on
  https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
  (MIT license)
  """
  dims = logits.dim()
  gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
  y = logits + gumbel_noise
  return F.softmax(y / T, dims - 1)


def gumbel_softmax(logits, k, T=1, hard=True, eps=1e-10):
  shape = logits.size()
  assert len(shape) == 2
  y_soft = _gumbel_softmax_sample(logits, T=T, eps=eps)
  if hard:
    _, ind = torch.topk(y_soft, k=k, dim=-1, largest=True)
    y_hard = logits.new_zeros(*shape).scatter_(-1, ind.view(-1, k), 1.0)
    y = y_hard - y_soft.detach() + y_soft
  else:
    y = y_soft
  return y