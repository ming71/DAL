# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights=(10., 10., 10., 5., 15.)):
        self.weights = weights

    def encode(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_thetas = ex_rois[:, 4]

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
        gt_thetas = gt_rois[:, 4]

        wx, wy, ww, wh, wt = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dt = wt * (torch.tan(gt_thetas / 180.0 * np.pi) - torch.tan(ex_thetas / 180.0 * np.pi))

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh, targets_dt), dim=1
        )
        return targets

    def decode(self, boxes, deltas, mode='xywht'):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights
        thetas = boxes[:, :, 4]

        wx, wy, ww, wh, wt = self.weights
        dx = deltas[:, :, 0] / wx
        dy = deltas[:, :, 1] / wy
        dw = deltas[:, :, 2] / ww
        dh = deltas[:, :, 3] / wh
        dt = deltas[:, :, 4] / wt

        pred_ctr_x = ctr_x if 'x' not in mode else ctr_x + dx * widths
        pred_ctr_y = ctr_y if 'y' not in mode else ctr_y + dy * heights
        pred_w = widths if 'w' not in mode else torch.exp(dw) * widths
        pred_h = heights if 'h' not in mode else torch.exp(dh) * heights
        pred_t = thetas if 't' not in mode else torch.atan(torch.tan(thetas / 180.0 * np.pi) + dt) / np.pi * 180.0

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2,
            pred_t], dim=2
        )
        return pred_boxes
