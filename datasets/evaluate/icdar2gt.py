import os 
import cv2
import numpy as np
import shutil
from decimal import Decimal
from tqdm import tqdm


def convert_detections(det_path):
    eval_dets = os.path.join(os.path.split(det_path)[0], 'detection-results')
    if os.path.exists(eval_dets):
        shutil.rmtree(eval_dets)
    os.mkdir(eval_dets)
    dets = os.listdir(det_path)
    for det in tqdm(dets):
        with open(os.path.join(det_path, det), 'r',encoding="gbk",errors='ignore') as f:
            res = f.readlines()
            res = ''.join(['text 0.99 '+x.replace(',', ' ') for x in res])
            with open(os.path.join(eval_dets, det[4:]), 'w') as fd: 
                fd.write(res)

def convert_icdar_gt(gt_path):
    eval_gts = os.path.join(os.path.split(gt_path)[0], 'ground-truth')
    if os.path.exists(eval_gts):
        shutil.rmtree(eval_gts)
    os.mkdir(eval_gts)
    gts = os.listdir(gt_path)
    for gt in tqdm(gts):
        with open(os.path.join(gt_path, gt), 'r',encoding="gbk",errors='ignore') as f:
            res = f.readlines()
            diff = ['###' in x for x in res]    
            res = ''.join(['text ' + ' '.join(x.split(',')[:8]) + '\n' if not diff[i] \
                else 'text ' + ' '.join(x.split(',')[:8]) + ' difficult' + '\n'  \
                for i, x in enumerate(res)])
            with open(os.path.join(eval_gts, gt[3:]), 'w') as fd:
                fd.write(res)


# if __name__ == "__main__":
    
#     det_path = '/py/BoxesCascade/test/detetions'    
#     gt_path = '/py/BoxesCascade/ICDAR15/test' 

# #    convert_detections(det_path)
#     convert_icdar_gt(gt_path)

    

