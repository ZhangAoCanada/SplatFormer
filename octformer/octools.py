# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import ocnn
import numpy as np
from tqdm import tqdm
from thsolver import Solver

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')

def get_input_feature(octree):
    octree_feature = ocnn.modules.InputFeature("FP", True)
    data = octree_feature(octree)
    return data

def process_batch(batch):
    def points2octree(points, device='cuda'):
        octree = ocnn.octree.Octree(11, 2, device=device)
        octree.build_octree(points)
        return octree

    if 'octree' in batch:
        batch['octree'] = batch['octree'].cuda(non_blocking=True)
        batch['points'] = batch['points'].cuda(non_blocking=True)
    else:
        points = batch["points"].cuda(non_blocking=True)
        octree = points2octree(points, device=points.device)
        octree.construct_all_neigh()
        batch['points'] = points
        batch['octree'] = octree
    return batch

def model_forward(batch, model):
    octree, points = batch['octree'], batch['points']
    data = get_input_feature(octree)
    query_pts = torch.cat([points.points, points.batch_id], dim=1)

    logit = model(data, octree, octree.depth, query_pts)
    # label_mask = points.labels > self.FLAGS.LOSS.mask  # filter labels
    # return logit[label_mask], points.labels[label_mask]
    return logit

def config_optimizer(model):
    base_lr = 0.0015
    transformer_lr_scale = 0.1
    parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if "blocks" not in n and p.requires_grad], },
        {"params": [p for n, p in model.named_parameters()
                    if "blocks" in n and p.requires_grad],
        "lr": base_lr * transformer_lr_scale, }, ]
    optimizer = torch.optim.AdamW(
        parameters, lr=base_lr, weight_decay=0.05)

def train_step(batch):
    batch = process_batch(batch)
    logit, label = model_forward(batch)
#     loss = loss_function(logit, label)
#     accu = accuracy(logit, label)
#     return {'train/loss': loss, 'train/accu': accu}

# def test_step(self, batch):
#     batch = self.process_batch(batch, self.FLAGS.DATA.test)
#     with torch.no_grad():
#         logit, label = self.model_forward(batch)
#     loss = self.loss_function(logit, label)
#     accu = self.accuracy(logit, label)
#     num_class = self.FLAGS.LOSS.num_class
#     IoU, insc, union = self.IoU_per_shape(logit, label, num_class)

#     names = ['test/loss', 'test/accu', 'test/mIoU'] + \
#             ['test/intsc_%d' % i for i in range(num_class)] + \
#             ['test/union_%d' % i for i in range(num_class)]
#     tensors = [loss, accu, IoU] + insc + union
#     return dict(zip(names, tensors))

# def eval_step(self, batch):
#     batch = self.process_batch(batch, self.FLAGS.DATA.test)
#     with torch.no_grad():
#         logit, _ = self.model_forward(batch)
#     prob = torch.nn.functional.softmax(logit, dim=1)

#     # split predictions
#     inbox_masks = batch['inbox_mask']
#     npts = batch['points'].batch_npt.tolist()
#     probs = torch.split(prob, npts)

#     # merge predictions
#     batch_size = len(inbox_masks)
#     for i in range(batch_size):
#         # The point cloud may be clipped when doing data augmentation. The
#         # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
#         # contains the prediction for all points.
#         prob = probs[i].cpu()
#         inbox_mask = inbox_masks[i].to(prob.device)
#         prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])
#         prob_all_pts[inbox_mask] = prob

#         # Aggregate predictions across different epochs
#         filename = batch['filename'][i]
#         self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

#         # Save the prediction results in the last epoch
#         if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
#             full_filename = os.path.join(self.logdir, filename[:-4] + '.eval.npz')
#             curr_folder = os.path.dirname(full_filename)
#             if not os.path.exists(curr_folder): os.makedirs(curr_folder)
#             np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

# def result_callback(self, avg_tracker, epoch):
#     r''' Calculate the part mIoU for PartNet and ScanNet.
#     '''

#     iou_part = 0.0
#     avg = avg_tracker.average()

#     # Labels smaller than `mask` is ignored. The points with the label 0 in
#     # PartNet are background points, i.e., unlabeled points
#     mask = self.FLAGS.LOSS.mask + 1
#     num_class = self.FLAGS.LOSS.num_class
#     for i in range(mask, num_class):
#         instc_i = avg['test/intsc_%d' % i]
#         union_i = avg['test/union_%d' % i]
#         iou_part += instc_i / (union_i + 1.0e-10)
#     iou_part = iou_part / (num_class - mask)

#     avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})
#     tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))

# def loss_function(self, logit, label):
#     criterion = torch.nn.CrossEntropyLoss()
#     loss = criterion(logit, label.long())
#     return loss

# def accuracy(self, logit, label):
#     pred = logit.argmax(dim=1)
#     accu = pred.eq(label).float().mean()
#     return accu

# def IoU_per_shape(self, logit, label, class_num):
#     pred = logit.argmax(dim=1)

#     IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
#     intsc, union = [None] * class_num, [None] * class_num
#     for k in range(class_num):
#         pk, lk = pred.eq(k), label.eq(k)
#         intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
#         union[k] = torch.sum(torch.logical_or(pk, lk).float())

#         valid = torch.sum(lk.any()) > 0
#         valid_part_num += valid.item()
#         IoU += valid * intsc[k] / (union[k] + esp)

#     # Calculate the shape IoU for ShapeNet
#     IoU /= valid_part_num + esp
#     return IoU, intsc, union
