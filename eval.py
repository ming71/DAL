from __future__ import print_function

import os
import cv2
import torch
import codecs
import zipfile
import shutil
import argparse
import sys
sys.path.append('datasets/DOTA_devkit')

from tqdm import tqdm
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_aabb, rbox_2_quad
from utils.utils import sort_corners, is_image, hyp_parse
from utils.map import eval_mAP

from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge
from datasets.DOTA_devkit.dota_evaluation_task1 import task1_eval


DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }

def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    # pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            # arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, filename)
    zipf.close()


def icdar_evaluate(model, 
                   target_size, 
                   test_path, 
                   dataset):
    if dataset == 'IC15':
        output = './datasets/IC_eval/icdar15'
    elif dataset == 'IC13':
        output = './datasets/IC_eval/icdar13'
    else:
        raise NotImplementedError

    ims_dir = test_path
    out_dir = './temp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size)
        nt += len(dets)
        out_file = os.path.join(out_dir, 'res_' + im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                continue
            if dataset == 'IC15':
                res = sort_corners(rbox_2_quad(dets[:, 2:]))
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                        res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                    )
            if dataset == 'IC13':
                res = rbox_2_aabb(dets[:, 2:])
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3])
                    )
    
    zip_name = 'submit.zip'
    make_zip(out_dir, zip_name)
    shutil.move(os.path.join('./', zip_name), os.path.join(output, zip_name))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    result = os.popen('cd {0} && python script.py -g=gt.zip -s=submit.zip '.format(output)).read()
    sep = result.split(':')
    precision = sep[1][:sep[1].find(',')].strip()
    recall = sep[2][:sep[2].find(',')].strip()
    f1 = sep[3][:sep[3].find(',')].strip()
    map = 0
    p = eval(precision)
    r = eval(recall)
    hmean = eval(f1)
    # display result
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', len(ims_list), nt, p, r, 0, hmean))
    return p, r, map, hmean 



def data_evaluate(model, 
                  target_size, 
                  test_path,
                  conf = 0.01,
                  dataset=None):
    root_dir = 'datasets/evaluate'
    out_dir = os.path.join(root_dir,'detection-results')
    if  os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    ds = DATASETS[dataset]()

    with open(test_path,'r') as f:
        if dataset == 'VOC':
            im_dir = test_path.replace('/ImageSets/Main/test.txt','/JPEGImages')
            ims_list = [os.path.join(im_dir, x.strip('\n')+'.jpg') for x in f.readlines()]
        else:
            ims_list = [x.strip('\n') for x in f.readlines() if is_image(x.strip('\n'))]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_path in enumerate(tqdm(ims_list, desc=s)):
        im_name = os.path.split(im_path)[1]
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf = conf)
        nt += len(dets)
        out_file = os.path.join(out_dir,  im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                    ds.return_class(dets[k, 0]), dets[k, 1],
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                )
        assert len(os.listdir(os.path.join(root_dir,'ground-truth'))) != 0, 'No labels found in test/ground-truth!! '
    mAP = eval_mAP(root_dir, use_07_metric=False)
    # display result
    pf = '%20s' + '%10.3g' * 6  # print format    
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    # thres = [0.5,0.6,0.7,0.8,0.9]
    # for thre in thres:
    #     ap = eval_mAP(root_dir, use_07_metric=True, thres = thre)
    #     print('ap_{}:  {}'.format(thre, ap))
    return 0, 0, mAP, 0 



def dota_evaluate(model, 
                  target_size, 
                  test_path,
                  conf = 0.01):
    # 
    root_data, evaldata = os.path.split(test_path)
    splitdata = evaldata + 'split'
    ims_dir = os.path.join(root_data, splitdata + '/' + 'images')
    root_dir = 'outputs'
    res_dir = os.path.join(root_dir, 'detections')          # 裁剪图像的检测结果   
    integrated_dir = os.path.join(root_dir, 'integrated')   # 将裁剪图像整合后成15个txt的结果
    merged_dir = os.path.join(root_dir, 'merged')           # 将整合后的结果NMS

    if  os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    for f in [res_dir, integrated_dir, merged_dir]: 
        if os.path.exists(f):
            shutil.rmtree(f)
        os.makedirs(f)

    ds = DOTADataset()
    # loss = torch.zeros(3)
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf = conf)
        nt += len(dets)
        out_file = os.path.join(res_dir,  im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {} {:.2f}\n'.format(
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                    ds.return_class(dets[k, 0]), im_name[:-4], dets[k, 1],)
                )
    ResultMerge(res_dir, integrated_dir, merged_dir)
    ## calc mAP
    mAP, classaps = task1_eval(merged_dir, test_path)
    # # display result
    pf = '%20s' + '%10.3g' * 6  # print format    
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    return 0, 0, mAP, 0 


def evaluate(target_size,
             test_path,
             dataset,
             backbone=None, 
             weight=None, 
             model=None,
             hyps=None,
             conf=0.3):
    if model is None:
        model = RetinaNet(backbone=backbone,hyps=hyps)
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)
            # load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    if 'IC' in dataset :
        results = icdar_evaluate(model, target_size, test_path, dataset)
    elif dataset in ['HRSC2016', 'UCAS_AOD', 'VOC', 'NWPU_VHR']:
        results = data_evaluate(model, target_size, test_path, conf, dataset)
    elif dataset == 'DOTA':
        results = dota_evaluate(model, target_size, test_path, conf)
    else:
        raise RuntimeError('Unsupported dataset!')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', dest='backbone', default='res50', type=str)
    parser.add_argument('--weight', type=str, default='weights/best.pth')
    parser.add_argument('--target_size', dest='target_size', default=[800], type=int) 
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')

    parser.add_argument('--dataset', nargs='?', type=str, default='NWPU_VHR')
    parser.add_argument('--test_path', type=str, default='NWPU_VHR/test.txt') 

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    evaluate(arg.target_size,
             arg.test_path,
             arg.dataset,
             arg.backbone,
             arg.weight,
             hyps = hyps)