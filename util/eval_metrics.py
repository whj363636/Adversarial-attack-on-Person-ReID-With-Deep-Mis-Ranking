from __future__ import print_function, absolute_import
import numpy as np
import torch
import copy
import os.path as osp
from collections import defaultdict
from opts import market1501_test_map, duke_test_map
import sys

def make_results(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids, targetmodel, ak_typ, attr_matrix=None, dataset_name=None, attr=None):
  qf, gf = featureNormalization(qf, gf, targetmodel)
  m, n = qf.size(0), gf.size(0)
  distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
  distmat.addmm_(1, -2, qf, gf.t())
  distmat = distmat.numpy()

  if targetmodel == 'aligned':
    from .distance import low_memory_local_dist
    lqf, lgf = lqf.permute(0,2,1), lgf.permute(0,2,1)
    local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(), aligned=True)
    distmat = local_distmat+distmat

  if ak_typ > 0: 
    distmat, all_hit, ignore_list = evaluate_attr(distmat, q_pids, g_pids, attr_matrix, dataset_name, attr)
    return distmat, all_hit, ignore_list
  else:
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)
    return distmat, cmc, mAP

def featureNormalization(qf, gf, targetmodel):
  if targetmodel in ['aligned', 'densenet121', 'hacnn', 'mudeep', 'ide', 'cam', 'lsro', 'hhl', 'spgan']:
    qf = 1. * qf / (torch.norm(qf, p=2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, p=2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)  

  elif targetmodel in ['pcb']:
    qf = (qf / (np.sqrt(6) * torch.norm(qf, p=2, dim=1, keepdim=True).expand_as(qf))).view(qf.size(0), -1)
    gf = (gf / (np.sqrt(6) * torch.norm(gf, p=2, dim=1, keepdim=True).expand_as(gf))).view(gf.size(0), -1)

  return qf, gf

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, use_metric_cuhk03=False):
  if use_metric_cuhk03: return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
  else: return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

def evaluate_attr(distmat, q_pids, g_pids, attr_matrix, dataset_name, attr_list, max_rank=20):
  attr_key, attr_value = attr_list
  attr_name = 'duke_attribute' if dataset_name == 'dukemtmcreid' else 'market_attribute'
  offset = 0 if dataset_name == 'dukemtmcreid' else 1
  mapping = duke_test_map if dataset_name == 'dukemtmcreid' else market1501_test_map
  column = attr_matrix[attr_name][0]['test'][0][0][attr_key][0][0]

  num_q, num_g = distmat.shape
  indices = np.argsort(distmat, axis=1)

  all_hit = []
  ignore_list = []
  num_valid_q = 0. # number of valid query
  for q_idx in range(num_q):
    q_pid = q_pids[q_idx]
    if column[mapping[q_pid]-offset] == attr_value: 
      ignore_list.append(q_idx)
      continue

    order = indices[q_idx]
    matches = np.zeros_like(order)
  
    for i in range(len(order)):
      if column[mapping[g_pids[order[i]]]-offset] == attr_value:
        matches[i] = 1

    hit = matches.cumsum()
    hit[hit > 1] = 1
    all_hit.append(hit[:max_rank])
    num_valid_q += 1. # number of valid query

  assert num_valid_q > 0
  all_hit = np.asarray(all_hit).astype(np.float32)
  all_hit = all_hit.sum(0) / num_valid_q

  # distmat = np.delete(distmat, ignore_list, axis=0)

  return distmat, all_hit, ignore_list