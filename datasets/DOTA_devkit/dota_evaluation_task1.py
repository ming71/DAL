# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import xml.etree.ElementTree as ET
import os
import cv2
import math
import codecs

#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial


# this function can be used for polygon
def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors

    vector_0 = np.array(bl-tl)
    vector_1 = np.array(rightMost[0]-tl)
    vector_2 = np.array(rightMost[1]-tl)

    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
    

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    # print(a, b)
    # print(zip(a, b))
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        #if i % 100 == 0:
         #   print ('Reading annotation for {:d}/{:d}'.format(
          #      i + 1, len(imagenames)) )
        # save
        #print ('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'w') as f:
         #   cPickle.dump(recs, f)
    #else:
        # load
        #with open(cachefile, 'r') as f:
         #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # import ipdb;ipdb.set_trace(context = 15)
    
    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])


    #print('check confidence: ', confidence)
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    if len(sorted_ind) != 0:
        BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni
            # 这里为止计算的是正框的IoU，用来剔除部分框加速计算（认为正框IoU为0说明riou也为0）
            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)

    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def eval_map(detpath,annopath,imagesetfile, use_07_metric=False):

    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    classaps = []
    map = 0
    for classname in classnames:
        # print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=use_07_metric)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
#         print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        plt.figure(figsize=(8,4))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec)
        plt.show()
        plt.savefig('PR-curve')
    map = map/len(classnames)
    # print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    return map, classaps

def generate_imageset(img_folder, imagesetfile):
    files = os.listdir(img_folder)
    with open(imagesetfile,'w') as f:
        for file in files:
            filename, extension = os.path.splitext(file)
            if extension in ['.jpg', '.bmp','.png','.tif','.tiff']:
                f.write(filename + '\n')

# dets和annos都是15txt大文件
'''
康康这个evaluate是怎么做的：
detpath是meige后大图上的15个txt结果的文件夹，但是注意最后用Task1_{:s}.txt，或者{:s}.txt输入全部txt文件，否则报文件夹的错
annopath是dataset直接给的，每个大图一个label的那个文件夹路径，同样最后注意加{:s}.txt索引txt文件
imagesetfile每行是一张图像的名称，如P1000.png，生成所有待测试的图像名即可（已提供函数实现）

note：
- 源程序在某类的det全为空时会报错，我改了下加了个判定，记录一下如果出问题可以改回，参见line179-180
'''

def task1_eval(detpath, val_folder,use_07_metric=False):
    img_dir = os.path.join(val_folder, 'images')
    imagesetfile = os.path.join(val_folder, 'imageset.txt')
    generate_imageset(img_dir, imagesetfile)
    
    detpath =  detpath + '/' + 'Task1_{:s}.txt'
    annopath = os.path.join(val_folder, 'labelTxt') + '/' + '{:s}.txt'
    map, classaps = eval_map(
                        detpath,
                        annopath,
                        imagesetfile,
                        use_07_metric
                    )
    return map, classaps

############ eval for HRSC dataset ####################
def xml2txt(root, src, dst, imageset_file):
    xml_dir = os.path.join(root, src)
    txt_dir = os.path.join(root, dst)
    imageset = os.path.join(root, imageset_file)
    os.mkdir(txt_dir)
    with open(imageset, 'r') as fs:
        filenames = [x.strip('\n') for x in fs.readlines()]
    for i, filename in enumerate(filenames):
        with open(os.path.join(xml_dir, filename + '.xml'), 'r') as f:
            content = f.read()
            content_splt = [x for x in content.split('<HRSC_Object>')[1:] if x!='']
            count = len(content_splt)
            out_file = os.path.join(txt_dir, filenames[i] + '.txt')
            with codecs.open(out_file, 'w', 'utf-8') as f:
                if count > 0:
                    s = ''
                    for obj in content_splt:
                        classname = 'ship'
                        class_ID = obj[obj.find('<Class_ID>')+10 : obj.find('</Class_ID>')]
                        difficult = obj[obj.find('<difficult>')+11 : obj.find('</difficult>')]
                        if class_ID in ['100000027', '100000022']:   # follow HRSC original settings
                            difficult = '1'
                        cx = float(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')])
                        cy = float(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')])
                        w  = float(obj[obj.find('<mbox_w>')+8 : obj.find('</mbox_w>')])
                        h  = float(obj[obj.find('<mbox_h>')+8 : obj.find('</mbox_h>')])
                        a  = obj[obj.find('<mbox_ang>')+10 : obj.find('</mbox_ang>')]
                        a = float(a) if not a[0]=='-' else -float(a[1:])
                        theta = a*180/math.pi
                        points = cv2.boxPoints(((cx,cy),(w,h),theta))
                        s += '{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {}\n'.format(
                                points[0][0], points[0][1], points[1][0], points[1][1],
                                points[2][0], points[2][1], points[3][0], points[3][1],
                                classname, difficult)
                    f.write(s)


def eval_HRSCDataset(detpath, val_folder):

    classnames = ['ship']
    if not os.path.exists(os.path.join(val_folder, 'labelTxt')):
        xml2txt(val_folder, 'FullDataSet/Annotations', 'labelTxt', 'ImageSets/test.txt')
    
    img_dir = os.path.join(val_folder, 'FullDataSet/AllImages')
    imagesetfile = os.path.join(val_folder, 'ImageSets/test.txt')
    
    detpath =  detpath + '/' + '{:s}.txt'
    annopath = os.path.join(val_folder, 'labelTxt') + '/' + '{:s}.txt'

    classaps = []
    map = 0
    for classname in classnames:
        # print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
#         print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        plt.figure(figsize=(8,4))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec)
        plt.show()
        plt.savefig('PR-curve')
    map = map/len(classnames)
    # print('map:', map)
    classaps = 100*np.array(classaps)
    # print('classaps: ', classaps)
    return map, classaps

############ eval for UCAS_AOD dataset ####################
def txt_trans(root, src, test, imageset_file):
    src_dir = os.path.join(root, src)
    test_dir = os.path.join(root, test)
    imageset = os.path.join(root, imageset_file)
    os.mkdir(test_dir)
    with open(imageset, 'r') as fs:
        filenames = [x.strip('\n') for x in fs.readlines()]
    for i, filename in enumerate(filenames):
        with open(os.path.join(src_dir, filename + '.txt'), 'r') as f:
            out_file = os.path.join(test_dir, filenames[i] + '.txt')
            with codecs.open(out_file, 'w', 'utf-8') as fw:
                lines = f.readlines()
                s = ''
                for line in lines:
                    classname, x1,y1,x2,y2,x3,y3,x4,y4, *_ = line.split()
                    x1,y1,x2,y2,x3,y3,x4,y4 = [eval(x) for x in (x1,y1,x2,y2,x3,y3,x4,y4)]
                    difficult = 0
                    s += '{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {}\n'.format(
                            x1,y1,x2,y2,x3,y3,x4,y4,classname, difficult)
                fw.write(s)

def eval_UCAS_AODDataset(detpath, val_folder):

    classnames = ['car', 'airplane']
    if not os.path.exists(os.path.join(val_folder, 'labelTxt')):
        txt_trans(val_folder, 'Annotations', 'labelTxt', 'ImageSets/test.txt')

    img_dir = os.path.join(val_folder, 'AllImages')
    imagesetfile = os.path.join(val_folder, 'ImageSets/test.txt')
    
    detpath =  detpath + '/' + '{:s}.txt'
    annopath = os.path.join(val_folder, 'labelTxt') + '/' + '{:s}.txt'
    classaps = []
    map = 0
    for classname in classnames:
        # print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
#         print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        plt.figure(figsize=(8,4))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec)
        plt.show()
        plt.savefig('PR-curve')
    map = map/len(classnames)
    # print('map:', map)
    classaps = 100*np.array(classaps)
    # print('classaps: ', classaps)
    return map, classaps


if __name__ == "__main__":
    
    detpath = r'/data-tmp/stela-master/datasets/DOTA_devkit/Task1_merge'
    val_folder = r'/data-tmp/stela-master/datasets/DOTA_devkit/example' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset

    map, classaps = task1_eval(detpath, val_folder)
    print(map, classaps)