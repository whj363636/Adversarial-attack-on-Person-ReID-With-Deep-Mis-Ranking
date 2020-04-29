import numpy as np
import os
import sys
import math
import random
import glob
import cv2
import torch
from scipy import io
from opts import market1501_train_map, duke_train_map, get_opts

market_dict = {'age':[1,2,3,4], # young(1), teenager(2), adult(3), old(4) 
               'backpack':[1,2], # no(1), yes(2)
               'bag':[1,2], # no(1), yes(2)
               'handbag':[1,2], # no(1), yes(2)
               'downblack':[1,2], # no(1), yes(2)
               'downblue':[1,2], # no(1), yes(2)
               'downbrown':[1,2], # no(1), yes(2)
               'downgray':[1,2], # no(1), yes(2)
               'downgreen':[1,2], # no(1), yes(2)
               'downpink':[1,2], # no(1), yes(2)
               'downpurple':[1,2], # no(1), yes(2)
               'downwhite':[1,2], # no(1), yes(2)
               'downyellow':[1,2], # no(1), yes(2)
               'upblack':[1,2], # no(1), yes(2)
               'upblue':[1,2], # no(1), yes(2)
               'upgreen':[1,2], # no(1), yes(2)
               'upgray':[1,2], # no(1), yes(2)
               'uppurple':[1,2], # no(1), yes(2)
               'upred':[1,2], # no(1), yes(2)
               'upwhite':[1,2], # no(1), yes(2)
               'upyellow':[1,2], # no(1), yes(2)
               'clothes':[1,2], # dress(1), pants(2)
               'down':[1,2], # long lower body clothing(1), short(2) 
               'up':[1,2], # long sleeve(1), short sleeve(2)
               'hair':[1,2], # short hair(1), long hair(2)
               'hat':[1,2], # no(1), yes(2)
               'gender':[1,2]}# male(1), female(2) 

duke_dict = {'gender':[1,2], # male(1), female(2) 
             'top':[1,2], # short upper body clothing(1), long(2)
             'boots':[1,2], # no(1), yes(2)
             'hat':[1,2], # no(1), yes(2)
             'backpack':[1,2], # no(1), yes(2)
             'bag':[1,2], # no(1), yes(2)
             'handbag':[1,2], # no(1), yes(2)
             'shoes':[1,2], # dark(1), light(2)
             'downblack':[1,2], # no(1), yes(2)
             'downwhite':[1,2], # no(1), yes(2)
             'downred':[1,2], # no(1), yes(2)
             'downgray':[1,2], # no(1), yes(2)
             'downblue':[1,2], # no(1), yes(2)
             'downgreen':[1,2], # no(1), yes(2)
             'downbrown':[1,2], # no(1), yes(2)
             'upblack':[1,2], # no(1), yes(2)
             'upwhite':[1,2], # no(1), yes(2)
             'upred':[1,2], # no(1), yes(2)
             'uppurple':[1,2], # no(1), yes(2)
             'upgray':[1,2], # no(1), yes(2)
             'upblue':[1,2], # no(1), yes(2)
             'upgreen':[1,2], # no(1), yes(2)
             'upbrown':[1,2]} # no(1), yes(2)

__dict_factory={
  'market_attribute': market_dict,
  'dukemtmcreid_attribute': duke_dict
}

def get_keys(dict_name):
  for key, value in __dict_factory.items():
    if key == dict_name:
      return value.keys()

def get_target_withattr(attr_matrix, dataset_name, attr_list, pids, pids_raw):
  attr_key, attr_value = attr_list
  attr_name = 'duke_attribute' if dataset_name == 'dukemtmcreid' else 'market_attribute'
  mapping = duke_train_map if dataset_name == 'dukemtmcreid' else market1501_train_map
  column = attr_matrix[attr_name][0]['train'][0][0][attr_key][0][0]

  n = pids_raw.size(0)
  targets = np.zeros_like(column)
  for i in range(n):
    if column[mapping[pids_raw[i].item()]] == attr_value:
      targets[pids[i].item()] = 1
  return torch.from_numpy(targets).view(1,-1).repeat(n, 1)