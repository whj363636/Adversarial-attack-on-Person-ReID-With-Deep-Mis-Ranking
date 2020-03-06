from __future__ import absolute_import
from collections import defaultdict
import numpy as np
import os.path as osp
import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, tp in enumerate(data_source):
            if len(tp) == 3:
                _, pid, _ = tp
            elif len(tp) == 4:
                _, pid, _, _ = tp

            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

class RandomIdentitySamplerCls(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, target) in enumerate(data_source):
            self.index_dic[target].append(index)
        self.targets = list(self.index_dic.keys())
        self.num_identities = len(self.targets)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            target = self.targets[i]
            t = self.index_dic[target]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

class AttrPool(Sampler):
  def __init__(self, data_source, dataset_name, attr_matrix, attr_list, sample_num):
    from opts import market1501_train_map, duke_train_map
    attr_key, attr_value = attr_list
    attr_name = 'duke_attribute' if dataset_name == 'dukemtmcreid' else 'market_attribute'
    mapping = duke_train_map if dataset_name == 'dukemtmcreid' else market1501_train_map
    column = attr_matrix[attr_name][0]['train'][0][0][attr_key][0][0]

    self.data_source = data_source
    self.sample_num = sample_num
    self.attr_pool = defaultdict(list)

    for index, (_, pid, _, pid_raw) in enumerate(data_source):
      if column[mapping[pid_raw]] == attr_value: 
        self.attr_pool[0].append(index)
      else:
        self.attr_pool[1].append(index)
    self.attrs = list(self.attr_pool.keys())
    self.num_attrs = len(self.attrs)

  def __iter__(self):
    ret = []
    for i in range(700):
      t = self.attr_pool[self.attrs[i%2]]
      replace = False if len(t) >= self.sample_num else True
      t = np.random.choice(t, size=self.sample_num, replace=replace)
      ret.extend(t)
    return iter(ret)

  def __len__(self):
    return self.sample_num*700