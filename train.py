from __future__ import absolute_import
from __future__ import print_function, division
import sys
import time
import datetime
import argparse
import os
import numpy as np
import os.path as osp
import math
from random import sample 
from scipy import io

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import models
from models.PCB import PCB_test
from ReID_attr import get_target_withattr
from opts import get_opts, Imagenet_mean, Imagenet_stddev
from GD import Generator, MS_Discriminator, Pat_Discriminator, GANLoss, weights_init
from advloss import DeepSupervision, adv_CrossEntropyLoss, adv_CrossEntropyLabelSmooth, adv_TripletLoss
from util import data_manager
from util.dataset_loader import ImageDataset
from util.utils import fliplr, Logger, save_checkpoint, visualize_ranked_results
from util.eval_metrics import make_results
from util.samplers import RandomIdentitySampler, AttrPool

# Training settings
parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('--targetmodel', type=str, default='aligned', choices=models.get_names())
parser.add_argument('--dataset', type=str, default='market1501', choices=data_manager.get_names())
# PATH
parser.add_argument('--G_resume_dir', type=str, default='', metavar='path to resume G')
parser.add_argument('--pre_dir', type=str, default='models', help='path to be attacked model')
parser.add_argument('--attr_dir', type=str, default='', help='path to attribute file')
parser.add_argument('--save_dir', type=str, default='logs', help='path to save model')
parser.add_argument('--vis_dir', type=str, default='vis', help='path to save visualization result')
parser.add_argument('--ablation', type=str, default='', help='for ablation study')
# var
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--D', type=str, default='MSGAN', help='Type of discriminator: PatchGAN or Multi-stage GAN')
parser.add_argument('--normalization', type=str, default='bn', help='bn or in')
parser.add_argument('--loss', type=str, default='xent_htri', choices=['cent', 'xent', 'htri', 'xent_htri'])
parser.add_argument('--ak_type', type=int, default=-1, help='-1 if non-targeted, 1 if attribute attack')
parser.add_argument('--attr_key', type=str, default='upwhite', help='[attribute, value]')
parser.add_argument('--attr_value', type=int, default=2, help='[attribute, value]')
parser.add_argument('--mag_in', type=float, default=16.0, help='l_inf magnitude of perturbation')
parser.add_argument('--temperature', type=float, default=-1, help="temperature for D")
parser.add_argument('--usegumbel', action='store_true', default=False, help='whether to use gumbel softmax')
parser.add_argument('--use_SSIM', type=int, default=2, help="0: None, 1: SSIM, 2: MS-SSIM ")
# Base
parser.add_argument('--train_batch', default=32, type=int,help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="test batch size")
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')

parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num_ker', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--print_freq', type=int, default=20, help="print frequency")
parser.add_argument('--eval_freq', type=int, default=1, help="eval frequency")
parser.add_argument('--usevis', action='store_true', default=False, help='whether to save vis')

args = parser.parse_args()
is_training = args.mode == 'train'
attr_list = [args.attr_key, args.attr_value]
attr_matrix = None
if args.attr_dir: 
  assert args.dataset in ['dukemtmcreid', 'market1501']
  attr_matrix = io.loadmat(args.attr_dir)
  args.ablation = osp.join('attr', args.attr_key + '=' + str(args.attr_value))

pre_dir = osp.join(args.pre_dir, args.targetmodel, args.dataset+'.pth.tar')
save_dir = osp.join(args.save_dir, args.targetmodel, args.dataset, args.ablation)
vis_dir = osp.join(args.vis_dir, args.targetmodel, args.dataset, args.ablation)


def main(opt):
  if not osp.exists(save_dir): os.makedirs(save_dir)
  if not osp.exists(vis_dir): os.makedirs(vis_dir)

  use_gpu = torch.cuda.is_available()
  pin_memory = True if use_gpu else False

  if args.mode == 'train': 
    sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))
  else: 
    sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))
  print("==========\nArgs:{}\n==========".format(args))

  if use_gpu:
    print("GPU mode")
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
  else:
    print("CPU mode")

  ### Setup dataset loader ###
  print("Initializing dataset {}".format(args.dataset))
  dataset = data_manager.init_img_dataset(root=args.root, name=args.dataset, split_id=opt['split_id'], cuhk03_labeled=opt['cuhk03_labeled'], cuhk03_classic_split=opt['cuhk03_classic_split'])
  if args.ak_type < 0:
    trainloader = DataLoader(ImageDataset(dataset.train, transform=opt['transform_train']), sampler=RandomIdentitySampler(dataset.train, num_instances=opt['num_instances']), batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
  elif args.ak_type > 0:
    trainloader = DataLoader(ImageDataset(dataset.train, transform=opt['transform_train']), sampler=AttrPool(dataset.train, args.dataset, attr_matrix, attr_list, sample_num=16), batch_size=args.train_batch, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=True)
  queryloader = DataLoader(ImageDataset(dataset.query, transform=opt['transform_test']), batch_size=args.test_batch, shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
  galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=opt['transform_test']), batch_size=args.test_batch, shuffle=False, num_workers=opt['workers'], pin_memory=pin_memory, drop_last=False)
  
  ### Prepare criterion ###
  if args.ak_type<0:
    clf_criterion = adv_CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu) if args.loss in ['xent', 'xent_htri'] else adv_CrossEntropyLoss(use_gpu=use_gpu)
  else:
    clf_criterion = nn.MultiLabelSoftMarginLoss()
  metric_criterion = adv_TripletLoss(margin=args.margin, ak_type=args.ak_type)
  criterionGAN = GANLoss()   

  ### Prepare pretrained model ###
  target_net = models.init_model(name=args.targetmodel, pre_dir=pre_dir, num_classes=dataset.num_train_pids)
  check_freezen(target_net, need_modified=True, after_modified=False)

  ### Prepare main net ###
  G = Generator(3, 3, args.num_ker, norm=args.normalization).apply(weights_init)
  if args.D == 'PatchGAN':
    D = Pat_Discriminator(input_nc=6, norm=args.normalization).apply(weights_init)
  elif args.D == 'MSGAN':
    D = MS_Discriminator(input_nc=6, norm=args.normalization, temperature=args.temperature, use_gumbel=args.usegumbel).apply(weights_init)
  check_freezen(G, need_modified=True, after_modified=True)
  check_freezen(D, need_modified=True, after_modified=True)
  print("Model size: {:.5f}M".format((sum(g.numel() for g in G.parameters())+sum(d.numel() for d in D.parameters()))/1000000.0))
  # setup optimizer
  optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
  optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
  
  if use_gpu: 
    test_target_net = nn.DataParallel(target_net).cuda() if not args.targetmodel == 'pcb' else nn.DataParallel(PCB_test(target_net)).cuda()
    target_net = nn.DataParallel(target_net).cuda() 
    G = nn.DataParallel(G).cuda()
    D = nn.DataParallel(D).cuda()

  if args.mode == 'test':
    epoch = 'test'
    test(G, D, test_target_net, dataset, queryloader, galleryloader, epoch, use_gpu, is_test=True)
    return 0

  # Ready
  start_time = time.time()
  train_time = 0
  worst_mAP, worst_rank1, worst_rank5, worst_rank10, worst_epoch = np.inf, np.inf, np.inf, np.inf, 0
  best_hit, best_epoch = -np.inf, 0
  print("==> Start training")

  for epoch in range(1,args.epoch+1):
    start_train_time = time.time()
    train(epoch, G, D, target_net, criterionGAN, clf_criterion, metric_criterion, optimizer_G, optimizer_D, trainloader, use_gpu)
    train_time += round(time.time() - start_train_time)

    if epoch % args.eval_freq == 0:
      print("==> Eval at epoch {}".format(epoch))
      if args.ak_type < 0:
        cmc, mAP = test(G, D, test_target_net, dataset, queryloader, galleryloader, epoch, use_gpu, is_test=False)
        is_worst = cmc[0]<=worst_rank1 and cmc[1]<=worst_rank5 and cmc[2]<=worst_rank10 and mAP<=worst_mAP
        if is_worst: 
          worst_mAP, worst_rank1, worst_epoch = mAP, cmc[0], epoch
        print("==> Worst_epoch is {}, Worst mAP {:.1%}, Worst rank-1 {:.1%}".format(worst_epoch, worst_mAP, worst_rank1))
        save_checkpoint(G.state_dict(), is_worst, 'G', osp.join(save_dir, 'G_ep' + str(epoch) + '.pth.tar'))
        save_checkpoint(D.state_dict(), is_worst, 'D', osp.join(save_dir, 'D_ep' + str(epoch) + '.pth.tar'))

      else: 
        all_hits = test(G, D, target_net, dataset, queryloader, galleryloader, epoch, use_gpu, is_test=False)
        is_best = all_hits[0]>=best_hit
        if is_best: 
          best_hit, best_epoch = all_hits[0], epoch
        print("==> Best_epoch is {}, Best rank-1 {:.1%}".format(best_epoch, best_hit))
        save_checkpoint(G.state_dict(), is_best, 'G', osp.join(save_dir, 'G_ep' + str(epoch) + '.pth.tar'))
        save_checkpoint(D.state_dict(), is_best, 'D', osp.join(save_dir, 'D_ep' + str(epoch) + '.pth.tar'))

  elapsed = round(time.time() - start_time)
  elapsed = str(datetime.timedelta(seconds=elapsed))
  train_time = str(datetime.timedelta(seconds=train_time))
  print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, G, D, target_net, criterionGAN, clf_criterion, metric_criterion, optimizer_G, optimizer_D, trainloader, use_gpu):
  G.train()
  D.train()
  global is_training
  is_training = True

  for batch_idx, (imgs, pids, _, pids_raw) in enumerate(trainloader):
    if use_gpu: 
      imgs, pids, pids_raw = imgs.cuda(), pids.cuda(), pids_raw.cuda()

    new_imgs, mask = perturb(imgs, G, D, train_or_test='train')
    new_imgs = new_imgs.cuda()
    mask = mask.cuda()
    # Fake Detection and Loss
    pred_fake_pool, _ = D(torch.cat((imgs, new_imgs.detach()), 1))
    loss_D_fake = criterionGAN(pred_fake_pool, False)        

    # Real Detection and Loss
    num = args.train_batch//2
    pred_real, _ = D(torch.cat((imgs[0:num,:,:,:], imgs[num:,:,:,:].detach()), 1))
    loss_D_real = criterionGAN(pred_real, True)

    # GAN loss (Fake Passability Loss)
    pred_fake, _ = D(torch.cat((imgs, new_imgs), 1))        
    loss_G_GAN = criterionGAN(pred_fake, True)               
    
    # Re-ID advloss
    ls = target_net(new_imgs, is_training)
    if len(ls) == 1: new_outputs = ls[0]
    if len(ls) == 2: new_outputs, new_features = ls
    if len(ls) == 3: new_outputs, new_features, new_local_features = ls
    xent_loss, global_loss, loss_G_ssim = 0, 0, 0
    targets = None

    if args.loss in ['cent', 'xent', 'xent_htri']:
      if args.ak_type < 0:
        xent_loss = DeepSupervision(clf_criterion, new_outputs, pids) if isinstance(new_features, (tuple, list)) else clf_criterion(new_outputs, pids)

      elif args.ak_type > 0:
        targets = get_target_withattr(attr_matrix, args.dataset, attr_list, pids, pids_raw).float().cuda()
        xent_loss = 0#DeepSupervision(clf_criterion, new_outputs, targets) if isinstance(new_features, (tuple, list)) else clf_criterion(new_outputs, targets)

    if args.loss in ['htri', 'xent_htri']:
      assert len(ls) >= 2 
      global_loss = DeepSupervision(metric_criterion, new_features, pids, targets) if isinstance(new_features, (tuple, list)) else metric_criterion(new_features, pids, targets)
    
    loss_G_ReID = (xent_loss+ global_loss)*opt['ReID_factor'] 

    # # SSIM loss
    if not args.use_SSIM == 0:
      from util.ms_ssim import msssim, ssim
      loss_func = msssim if args.use_SSIM == 2 else ssim
      loss_G_ssim = (1-loss_func(imgs, new_imgs))*0.1

    ############## Forward ###############
    loss_D = (loss_D_fake + loss_D_real)/2
    loss_G = loss_G_GAN + loss_G_ReID + loss_G_ssim
    ############## Backward #############
    # update generator weights
    optimizer_G.zero_grad()
    # loss_G.backward(retain_graph=True)
    loss_G.backward()
    optimizer_G.step()
    # update discriminator weights
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
    if (batch_idx+1) % args.print_freq == 0:
      print("===> Epoch[{}]({}/{}) loss_D: {:.4f} loss_G_GAN: {:.4f} loss_G_ReID: {:.4f} loss_G_SSIM: {:.4f}".format(epoch, batch_idx, len(trainloader), loss_D.item(), loss_G_GAN.item(), loss_G_ReID.item(), loss_G_ssim))

def test(G, D, target_net, dataset, queryloader, galleryloader, epoch, use_gpu, is_test=False, ranks=[1, 5, 10, 20]):
  global is_training
  is_training = False
  if args.mode == 'test' and args.G_resume_dir:
    G_resume_dir, D_resume_dir = args.G_resume_dir, args.G_resume_dir.replace('G', 'D')
    G_checkpoint, D_checkpoint = torch.load(G_resume_dir), torch.load(D_resume_dir)
    G_state_dict = G_checkpoint['state_dict'] if isinstance(G_checkpoint, dict) and 'state_dict' in G_checkpoint else G_checkpoint
    D_state_dict = D_checkpoint['state_dict'] if isinstance(D_checkpoint, dict) and 'state_dict' in D_checkpoint else D_checkpoint

    G.load_state_dict(G_state_dict)
    D.load_state_dict(D_state_dict)
    print("Sucessfully, loading {} and {}".format(G_resume_dir, D_resume_dir))

  with torch.no_grad():
    qf, lqf, new_qf, new_lqf, q_pids, q_camids = extract_and_perturb(queryloader, G, D, target_net, use_gpu, query_or_gallery='query', is_test=is_test, epoch=epoch)
    gf, lgf, g_pids, g_camids = extract_and_perturb(galleryloader, G, D, target_net, use_gpu, query_or_gallery='gallery', is_test=is_test, epoch=epoch)

    if args.ak_type > 0:
      distmat, hits, ignore_list = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type, attr_matrix, args.dataset, attr_list)
      print("Hits rate, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(ranks[0], hits[ranks[0]-1], ranks[1], hits[ranks[1]-1], ranks[2], hits[ranks[2]-1], ranks[3], hits[ranks[3]-1]))
      if not is_test:
        return hits

    else:
      if is_test:
        distmat, cmc, mAP = make_results(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
        new_distmat, new_cmc, new_mAP = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
        print("Results ----------")
        print("Before, mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(mAP, ranks[0], cmc[ranks[0]-1], ranks[1], cmc[ranks[1]-1], ranks[2], cmc[ranks[2]-1], ranks[3], cmc[ranks[3]-1]))
        print("After , mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_mAP, ranks[0], new_cmc[ranks[0]-1], ranks[1], new_cmc[ranks[1]-1], ranks[2], new_cmc[ranks[2]-1], ranks[3], new_cmc[ranks[3]-1]))
        # if args.usevis: 
        #   visualize_ranked_results(distmat, dataset, save_dir=osp.join(vis_dir, 'origin_results'), topk=20)
        # if args.usevis: 
        #   visualize_ranked_results(new_distmat, dataset, save_dir=osp.join(vis_dir, 'polluted_results'), topk=20)
      else:
        _, new_cmc, new_mAP = make_results(new_qf, gf, new_lqf, lgf, q_pids, g_pids, q_camids, g_camids, args.targetmodel, args.ak_type)
        print("mAP: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}, Rank-{}: {:.1%}".format(new_mAP, ranks[0], new_cmc[ranks[0]-1], ranks[1], new_cmc[ranks[1]-1], ranks[2], new_cmc[ranks[2]-1], ranks[3], new_cmc[ranks[3]-1]))
        return new_cmc, new_mAP

def extract_and_perturb(loader, G, D, target_net, use_gpu, query_or_gallery, is_test, epoch):
  f, lf, new_f, new_lf, l_pids, l_camids = [], [], [], [], [], []
  ave_mask, num = 0, 0
  for batch_idx, (imgs, pids, camids, pids_raw) in enumerate(loader):
    if use_gpu: 
      imgs = imgs.cuda()
    ls = extract(imgs, target_net)
    if len(ls) == 1: features = ls[0]
    if len(ls) == 2: 
      features, local_features = ls
      lf.append(local_features.detach().data.cpu())

    f.append(features.detach().data.cpu())
    l_pids.extend(pids)
    l_camids.extend(camids)

    if query_or_gallery == 'query':
      G.eval()
      D.eval()
      new_imgs, delta, mask = perturb(imgs, G, D, train_or_test='test')
      ave_mask += torch.sum(mask.detach()).cpu().numpy()
      num += imgs.size(0)

      ls = extract(new_imgs, target_net)
      if len(ls) == 1: new_features = ls[0]
      if len(ls) == 2: 
        new_features, new_local_features = ls
        new_lf.append(new_local_features.detach().data.cpu())
      new_f.append(new_features.detach().data.cpu())

      ls = [imgs, new_imgs, delta, mask]
      if is_test: 
        save_img(ls, pids, camids, epoch, batch_idx)

  f = torch.cat(f, 0)
  if not lf == []: lf = torch.cat(lf, 0)
  l_pids, l_camids = np.asarray(l_pids), np.asarray(l_camids)
  
  print("Extracted features for {} set, obtained {}-by-{} matrix".format(query_or_gallery, f.size(0), f.size(1)))
  if query_or_gallery == 'gallery':
    return [f, lf, l_pids, l_camids]
  elif query_or_gallery == 'query':
    new_f = torch.cat(new_f, 0)
    if not new_lf == []: 
      new_lf = torch.cat(new_lf, 0)
    return [f, lf, new_f, new_lf, l_pids, l_camids]

def extract(imgs, target_net):
  if args.targetmodel in ['pcb', 'lsro']:
    ls = [target_net(imgs, is_training)[0] + target_net(fliplr(imgs), is_training)[0]]
  else: 
    ls = target_net(imgs, is_training)
  for i in range(len(ls)): ls[i] = ls[i].data.cpu()
  return ls

def perturb(imgs, G, D, train_or_test='test'):
  n,c,h,w = imgs.size()
  delta = G(imgs)
  delta = L_norm(delta, train_or_test)
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
  delta = delta * mask
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())

  for c in range(3):
    new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp(new_imgs.data[:,c,:,:].min(), new_imgs.data[:,c,:,:].max()) # do clamping per channel
  if train_or_test == 'train':
    return new_imgs, mask
  elif train_or_test == 'test':
    return new_imgs, delta, mask

def L_norm(delta, mode='train'):
  delta.data += 1 
  delta.data *= 0.5

  for c in range(3):
    delta.data[:,c,:,:] = (delta.data[:,c,:,:] - Imagenet_mean[c]) / Imagenet_stddev[c]

  bs = args.train_batch if (mode == 'train') else args.test_batch
  for i in range(bs):
    # do per channel l_inf normalization
    for ci in range(3):
      try:
        l_inf_channel = delta[i,ci,:,:].data.abs().max()
        # l_inf_channel = torch.norm(delta[i,ci,:,:]).data
        mag_in_scaled_c = args.mag_in/(255.0*Imagenet_stddev[ci])
        delta[i,ci,:,:].data *= np.minimum(1.0, mag_in_scaled_c / l_inf_channel).float().cuda()
      except IndexError:
        break
  return delta

def save_img(ls, pids, camids, epoch, batch_idx):
  image, new_image, delta, mask = ls
  # undo normalize image color channels
  delta_tmp = torch.zeros(delta.size())
  for c in range(3):
    image.data[:,c,:,:] = (image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
    new_image.data[:,c,:,:] = (new_image.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]
    delta_tmp.data[:,c,:,:] = (delta.data[:,c,:,:] * Imagenet_stddev[c]) + Imagenet_mean[c]

  if args.usevis: 
    torchvision.utils.save_image(image.data, osp.join(vis_dir, 'original_epoch{}_batch{}.png'.format(epoch, batch_idx)))
    torchvision.utils.save_image(new_image.data, osp.join(vis_dir, 'polluted_epoch{}_batch{}.png'.format(epoch, batch_idx)))
    torchvision.utils.save_image(delta_tmp.data, osp.join(vis_dir, 'delta_epoch{}_batch{}.png'.format(epoch, batch_idx)))
    torchvision.utils.save_image(mask.data*255, osp.join(vis_dir, 'mask_epoch{}_batch{}.png'.format(epoch, batch_idx)))

def check_freezen(net, need_modified=False, after_modified=None):
  # print(net)
  cc = 0
  for child in net.children():
    for param in child.parameters():
      if need_modified: param.requires_grad = after_modified
      # if param.requires_grad: print('child', cc , 'was active')
      # else: print('child', cc , 'was forzen')
    cc += 1

if __name__ == '__main__':
  opt = get_opts(args.targetmodel)
  main(opt)