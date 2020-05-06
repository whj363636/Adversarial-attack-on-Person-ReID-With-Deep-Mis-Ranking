from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import errno
import shutil
import json
import time
import os.path as osp
from PIL import Image
import matplotlib
import numpy as np
from numpy import array, argmin

import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def save_heatmap(path, den):
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.colors import PowerNorm, LogNorm
  import matplotlib.cm as cm  
  plt.axis('off')
  plt.imshow(den, 
             cmap=cm.jet, 
             Norm=LogNorm(), 
             interpolation="bicubic")
  # save fig
  fig = plt.gcf()
  fig.savefig(path, format='png', bbox_inches='tight', transparent=True, dpi=600)
  plt.close('all')

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, G_or_D, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_'+ G_or_D +'.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def _traceback(D):
    i,j = array(D.shape)-1
    p,q = [i],[j]
    while (i>0) or (j>0):
        tb = argmin((D[i,j-1], D[i-1,j]))
        if tb == 0:
            j -= 1
        else: #(tb==1)
            i -= 1
        p.insert(0,i)
        q.insert(0,j)
    return array(p), array(q)

def dtw(dist_mat):
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    path = _traceback(dist)
    return dist[-1,-1]/sum(dist.shape), dist, path

def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will Redo. Don't worry. Just chill".format(img_path))
            pass
    return img

def img_to_tensor(img,transform):
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def feat_flatten(feat):
    shp = feat.shape
    feat = feat.reshape(shp[0] * shp[1], shp[2])
    return feat

def merge_feature(feature_list, shp, sample_rate = None):
    def pre_process(torch_feature_map):
        numpy_feature_map = torch_feature_map.cpu().data.numpy()[0]
        numpy_feature_map = numpy_feature_map.transpose(1,2,0)
        shp = numpy_feature_map.shape[:2]
        return numpy_feature_map, shp
    def resize_as(tfm, shp):
        nfm, shp2 = pre_process(tfm)
        scale = shp[0]/shp2[0]
        nfm1 = nfm.repeat(scale, axis = 0).repeat(scale, axis=1)
        return nfm1
    final_nfm = resize_as(feature_list[0], shp)
    for i in range(1, len(feature_list)):
        temp_nfm = resize_as(feature_list[i],shp)
        final_nfm = np.concatenate((final_nfm, temp_nfm),axis =-1)
    if sample_rate > 0:
        final_nfm = final_nfm[0:-1:sample_rate, 0:-1,sample_rate, :]
    return final_nfm

def visualize_ranked_results(distmat, dataset, save_dir, topk=20):
  """
  Visualize ranked results
  Support both imgreid and vidreid
  Args:
  - distmat: distance matrix of shape (num_query, num_gallery).
  - dataset: has dataset.query and dataset.gallery, both are lists of (img_path, pid, camid);
             for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
             a sequence of strings.
  - save_dir: directory to save output images.
  - topk: int, denoting top-k images in the rank list to be visualized.
  """
  num_q, num_g = distmat.shape

  print("Visualizing top-{} ranks in '{}' ...".format(topk, save_dir))
  print("# query: {}. # gallery {}".format(num_q, num_g))
  
  assert num_q == len(dataset.query)
  assert num_g == len(dataset.gallery)
  
  indices = np.argsort(distmat, axis=1)
  mkdir_if_missing(save_dir)

  for q_idx in range(num_q):
    qimg_path, qpid, qcamid = dataset.query[q_idx]
    qdir = osp.join(save_dir, 'query' + str(q_idx + 1).zfill(5))
    mkdir_if_missing(qdir)
    cp_img_to(qimg_path, qdir, rank=0, prefix='query')

    rank_idx = 1
    for g_idx in indices[q_idx,:]:
      gimg_path, gpid, gcamid = dataset.gallery[g_idx]
      invalid = (qpid == gpid) & (qcamid == gcamid)
      if not invalid:
        cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
        rank_idx += 1
        if rank_idx > topk:
            break

def cp_img_to(src, dst, rank, prefix):
    """
    - src: image path or tuple (for vidreid)
    - dst: target directory
    - rank: int, denoting ranked position, starting from 1
    - prefix: string
    """
    if isinstance(src, tuple) or isinstance(src, list):
      dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
      mkdir_if_missing(dst)
      for img_path in src:
        shutil.copy(img_path, dst)
    else:
      dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
      shutil.copy(src, dst)