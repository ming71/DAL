import numpy as np
import torch

from torchvision.transforms import Compose

from utils.utils import Rescale, Normailize, Reshape
from utils.nms_wrapper import nms


def im_detect(model, src, target_sizes, use_gpu=True, conf=None):
    if isinstance(target_sizes, int):
        target_sizes = [target_sizes]
    if len(target_sizes) == 1:
        return single_scale_detect(model, src, target_size=target_sizes[0], use_gpu=use_gpu, conf=conf)
    else:
        ms_dets = None
        for ind, scale in enumerate(target_sizes):
            cls_dets = single_scale_detect(model, src, target_size=scale, use_gpu=use_gpu, conf=conf)
            if cls_dets.shape[0] == 0:
                continue
            if ms_dets is None:
                ms_dets = cls_dets
            else:
                ms_dets = np.vstack((ms_dets, cls_dets))
        if ms_dets is None:
            return np.zeros((0, 7))
        cls_dets = np.hstack((ms_dets[:, 2:7], ms_dets[:, 1][:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, 0.1)
        return ms_dets[keep, :]


def single_scale_detect(model, src, target_size, use_gpu=True, conf=None):
    im, im_scales = Rescale(target_size=target_size, keep_ratio=True)(src)
    im = Compose([Normailize(), Reshape(unsqueeze=True)])(im)
    if use_gpu and torch.cuda.is_available():
        model, im = model.cuda(), im.cuda()
    with torch.no_grad():
        scores, classes, boxes = model(im, test_conf = conf)
    scores = scores.data.cpu().numpy()
    classes = classes.data.cpu().numpy()
    boxes = boxes.data.cpu().numpy()
    boxes[:, :4] = boxes[:, :4] / im_scales
    if boxes.shape[1] > 5:   
        boxes[:, 5:9] = boxes[:, 5:9] / im_scales
    scores = np.reshape(scores, (-1, 1))
    classes = np.reshape(classes, (-1, 1))
    cls_dets = np.concatenate([classes, scores, boxes], axis=1)
    keep = np.where(classes > 0)[0]
    return cls_dets[keep, :]
    # cls, score, x,y,x,y,a,   a_x,a_y,a_x,a_y,a_a  
