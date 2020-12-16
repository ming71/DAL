import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps
from utils.overlaps_cuda.rbbox_overlaps  import rbbx_overlaps


def xyxy2xywh_a(query_boxes):
    out_boxes = query_boxes.copy()
    out_boxes[:, 0] = (query_boxes[:, 0] + query_boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (query_boxes[:, 1] + query_boxes[:, 3]) * 0.5
    out_boxes[:, 2] = query_boxes[:, 2] - query_boxes[:, 0]
    out_boxes[:, 3] = query_boxes[:, 3] - query_boxes[:, 1]
    return out_boxes

# cuda_overlaps
class IntegratedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func = 'smooth'):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss

    def forward(self, classifications, regressions, anchors, refined_achors, annotations, \
                md_thres=0.5, mining_param=(1, 0., -1), ref=False):
        
        das = True
        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        alpha, beta, var = mining_param
#         import ipdb;ipdb.set_trace()
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            if bbox_annotation.shape[0] == 0:
                cls_losses.append(torch.tensor(0).float().cuda())
                reg_losses.append(torch.tensor(0).float().cuda())
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            sa = rbbx_overlaps(
                xyxy2xywh_a(anchors[j, :, :].cpu().numpy()),
                xyxy2xywh_a(bbox_annotation[:, :-1].cpu().numpy()),
            )
            if not torch.is_tensor(sa):
                # import ipdb;ipdb.set_trace()
                sa = torch.from_numpy(sa).cuda()
            if var != -1:
                fa = rbbx_overlaps(
                    xyxy2xywh_a(refined_achors[j, :, :].cpu().numpy()),
                    xyxy2xywh_a(bbox_annotation[:, :-1].cpu().numpy()),
                )
                if not torch.is_tensor(fa):
                    fa = torch.from_numpy(fa).cuda()

                if var == 0:
                    md = abs((alpha * sa + beta * fa))
                else:
                    md = abs((alpha * sa + beta * fa) - abs(fa - sa)**var)
            else:
                das = False
                md = sa
            
            iou_max, iou_argmax = torch.max(md, dim=1)
           
            positive_indices = torch.ge(iou_max, md_thres)

             
            max_gt, argmax_gt = md.max(0) 
            # import ipdb;ipdb.set_trace(context = 15)
            if (max_gt < md_thres).any():
                positive_indices[argmax_gt[max_gt < md_thres]]=1
              
            # matching-weight
            if das:
                pos = md[positive_indices]
                pos_mask =  torch.ge(pos, md_thres)
                max_pos, armmax_pos = pos.max(0)
                nt = md.shape[1]
                for gt_idx in range(nt):
                    pos_mask[armmax_pos[gt_idx], gt_idx] = 1
                comp = torch.where(pos_mask, (1 - max_pos).repeat(len(pos),1), pos)
                matching_weight = comp + pos
            # import ipdb; ipdb.set_trace(context = 15)

            # cls loss
            cls_targets = (torch.ones(classification.shape) * -1).cuda()
            cls_targets[torch.lt(iou_max, md_thres - 0.1), :] = 0
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[iou_argmax, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1
            alpha_factor = torch.ones(cls_targets.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bin_cross_entropy = -(cls_targets * torch.log(classification+1e-6) + (1.0 - cls_targets) * torch.log(1.0 - classification+1e-6))
            if das :
                soft_weight = (torch.zeros(classification.shape)).cuda()
                soft_weight = torch.where(torch.eq(cls_targets, 0.), torch.ones_like(cls_targets), soft_weight)
                soft_weight[positive_indices, assigned_annotations[positive_indices, -1].long()] = (matching_weight.max(1)[0] + 1)
                cls_loss = focal_weight * bin_cross_entropy * soft_weight
            else:
                cls_loss = focal_weight * bin_cross_entropy 
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            # reg loss
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)
                if das:
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets, weight = matching_weight)
                else:
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                reg_losses.append(reg_loss)

                if not torch.isfinite(reg_loss) :
                    import ipdb; ipdb.set_trace()
                k=1
            else:
                reg_losses.append(torch.tensor(0).float().cuda())
        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg = torch.stack(reg_losses).mean(dim=0, keepdim=True)
        return loss_cls, loss_reg

    
def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight = None):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    if  weight is  None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1)[0].unsqueeze(1).repeat(1,5)
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()

