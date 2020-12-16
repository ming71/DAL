import numpy as np
import cv2
import torch



def mask_valid_boxes(boxes, return_mask=False):
    """
    :param boxes: (cx, cy, w, h,*_) 
    :return: mask
    """   
    w = boxes[:,2]
    h = boxes[:,3]
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    mask = (w > 2) & (h > 2) & (ar < 30) 
    if return_mask:
        return mask
    else:
        return boxes[mask]



def xy2wh(boxes):
    """
    :param boxes: (xmin, ymin, xmax, ymax) (n, 4)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
    """
    if torch.is_tensor(boxes):
        out_boxes = boxes.clone()
    else:
        out_boxes = boxes.copy()
    out_boxes[:, 2] = boxes[:, 2] - boxes[:, 0] + 1.0
    out_boxes[:, 3] = boxes[:, 3] - boxes[:, 1] + 1.0
    out_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5

    return out_boxes

def constraint_theta(bboxes, mode = 'xywha'):
    keep_dim = False
    if len(bboxes.shape) == 1:
        keep_dim = True
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.unsqueeze(0) 
        if isinstance(bboxes,np.ndarray):
            bboxes = np.expand_dims(bboxes, 0) 
        else: bboxes = [bboxes]
    if isinstance(bboxes,torch.Tensor):
        assert bboxes.grad_fn is False, 'Modifying variables to be calc. grad. is not allowed!!'
    assert (bboxes[:, 4] >= -90).all() and (bboxes[:, 4] <= 90).all(), 'pleast restrict theta to (-90,90)'       
    for box in bboxes:
        if box[4] > 45.0:
            box[2], box[3] = box[3], box[2]
            box[4] -= 90
        elif box[4] < -45.0:
            box[2], box[3] = box[3], box[2]
            box[4] += 90
        elif abs(box[4]) == 45:
            if box[2] > box[3]:
                box[4] = -45
            else:
                box[4] = 45
                box[2], box[3] = box[3], box[2]
    if keep_dim:
        return bboxes[0]
    else :
        return bboxes



def bbox_overlaps(boxes, query_boxes):
    if not isinstance(boxes,float):   # apex
        boxes = boxes.float()
    area = (query_boxes[:, 2] - query_boxes[:, 0]) * \
           (query_boxes[:, 3] - query_boxes[:, 1])
    iw = torch.min(torch.unsqueeze(boxes[:, 2], dim=1), query_boxes[:, 2]) - \
         torch.max(torch.unsqueeze(boxes[:, 0], 1), query_boxes[:, 0])
    ih = torch.min(torch.unsqueeze(boxes[:, 3], dim=1), query_boxes[:, 3]) - \
         torch.max(torch.unsqueeze(boxes[:, 1], 1), query_boxes[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    return intersection / ua


def rbox_overlaps(boxes, query_boxes, indicator=None, thresh=1e-1):
    # rewrited by cython
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    a_tt = boxes[:, 4]
    a_ws = boxes[:, 2] - boxes[:, 0]
    a_hs = boxes[:, 3] - boxes[:, 1]
    a_xx = boxes[:, 0] + a_ws * 0.5
    a_yy = boxes[:, 1] + a_hs * 0.5

    b_tt = query_boxes[:, 4]
    b_ws = query_boxes[:, 2] - query_boxes[:, 0]
    b_hs = query_boxes[:, 3] - query_boxes[:, 1]
    b_xx = query_boxes[:, 0] + b_ws * 0.5
    b_yy = query_boxes[:, 1] + b_hs * 0.5

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = b_ws[k] * b_hs[k]
        for n in range(N):
            if indicator is not None and indicator[n, k] < thresh:
                continue
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                ((b_xx[k], b_yy[k]), (b_ws[k], b_hs[k]), b_tt[k])
            )
            if rtn == 1:
                ia = cv2.contourArea(contours)
                overlaps[n, k] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n, k] = ia / (ua - ia)
    return overlaps

def quad_2_rbox(quads, mode='xyxya'):
    # http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg
    if len(quads.shape) == 1:
        quads = quads[np.newaxis, :]
    rboxes = np.zeros((quads.shape[0], 5), dtype=np.float32)
    for i, quad in enumerate(quads):
        rbox = cv2.minAreaRect(quad.reshape([4, 2]))    
        x, y, w, h, t = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]
        if np.abs(t) < 45.0:
            rboxes[i, :] = np.array([x, y, w, h, t])
        elif np.abs(t) > 45.0:
            rboxes[i, :] = np.array([x, y, h, w, 90.0 + t])
        else:   
            if w > h:
                rboxes[i, :] = np.array([x, y, w, h, -45.0])
            else:
                rboxes[i, :] = np.array([x, y, h, w, 45])
    # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    if mode == 'xyxya':
        rboxes[:, 0:2] = rboxes[:, 0:2] - rboxes[:, 2:4] * 0.5
        rboxes[:, 2:4] = rboxes[:, 0:2] + rboxes[:, 2:4]
    rboxes[:, 0:4] = rboxes[:, 0:4].astype(np.int32)
    return rboxes

def rbox_2_quad(rboxes, mode='xyxya'):
    if len(rboxes.shape) == 1:
        rboxes = rboxes[np.newaxis, :]
    if rboxes.shape[0] == 0:
        return rboxes
    quads = np.zeros((rboxes.shape[0], 8), dtype=np.float32)
    for i, rbox in enumerate(rboxes):
        if len(rbox!=0):
            if mode == 'xyxya':
                w = rbox[2] - rbox[0]
                h = rbox[3] - rbox[1]
                x = rbox[0] + 0.5 * w
                y = rbox[1] + 0.5 * h
                theta = rbox[4]
            elif mode == 'xywha':
                x = rbox[0]
                y = rbox[1]
                w = rbox[2]
                h = rbox[3]
                theta = rbox[4]
            quads[i, :] = cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))

    return quads


def quad_2_aabb(quads):
    aabb = np.zeros((quads.shape[0], 4), dtype=np.float32)
    aabb[:, 0] = np.min(quads[:, 0::2], 1)
    aabb[:, 1] = np.min(quads[:, 1::2], 1)
    aabb[:, 2] = np.max(quads[:, 0::2], 1)
    aabb[:, 3] = np.max(quads[:, 1::2], 1)
    return aabb


def rbox_2_aabb(rboxes):
    if len(rboxes.shape) == 1:
        rboxes = rboxes[np.newaxis, :]
    if rboxes.shape[0] == 0:
        return rboxes
    quads = rbox_2_quad(rboxes)
    aabbs = quad_2_aabb(quads)
    return aabbs

# (num_boxes, 5)  xyxya
def min_area_square(rboxes):
    w = rboxes[:, 2] - rboxes[:, 0]
    h = rboxes[:, 3] - rboxes[:, 1]
    ctr_x = rboxes[:, 0] + w * 0.5
    ctr_y = rboxes[:, 1] + h * 0.5
    s = torch.max(w, h)
    return torch.stack((
        ctr_x - s * 0.5, ctr_y - s * 0.5,
        ctr_x + s * 0.5, ctr_y + s * 0.5),
        dim=1
    )


def clip_boxes(boxes, ims):
    _, _, h, w = ims.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=w)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=h)
    return boxes


if __name__ == '__main__':
    box = quad_2_rbox(np.array([[729.0, 708.0, 745.0 ,701.0, 758.0 ,731.0 ,744.0, 735.0]]),mode= 'xywha')
    print(box)
