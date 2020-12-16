# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Extended by Linjie Deng
# --------------------------------------------------------
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET

from utils.bbox import quad_2_rbox


class VOCDataset(data.Dataset):
    """"""
    def __init__(self,
                 dataset='trainval.txt',
                 augment = False,
                 level = 1,
                 random_flip=True):
        self.image_set = dataset
        self.data_path = self.image_set.strip('/ImageSets/Main/trainval.txt')
        self.image_ext = [".jpg"]
        self.image_list = self._load_image_names()
        self.classes = ('__background__', 'aeroplane','bicycle','bird','boat',
                        'bottle','bus','car','cat','chair','cow','diningtable',
                        'dog','horse','motorbike','person','pottedplant',
                        'sheep','sofa','train','tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_path = self._image_path_from_index(self.image_list[index])
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roidb = self._load_pascal_annotation(self.image_list[index])
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
        bboxes = roidb['boxes'][gt_inds, :]
        classes = roidb['gt_classes'][gt_inds]

        if self.random_flip and np.random.rand() >= 0.5:
            im = cv2.flip(im, 1, None)
            oldxs = bboxes[:, 0::2].copy()
            bboxes[:, 0::2] = im.shape[1] - oldxs - 1

        gt_boxes = np.empty((len(gt_inds), 6), dtype=np.float32)
        for i, bbox in enumerate(bboxes):
            gt_boxes[i, :5] = quad_2_rbox(np.array(bbox))
            gt_boxes[i, 5] = classes[i]
        return {'image': im, 'boxes': gt_boxes}

    def _load_image_names(self):
        """
        Load the names listed in this dataset's image set file.
        """
        image_set_file = self.image_set
        if not os.path.exists(image_set_file):
            'Path does not exist: {}'.format(image_set_file)
            image_names = []
        else:
            with open(image_set_file) as f:
                image_names = [x.strip() for x in f.readlines()]
        return image_names

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = None
        image_exist = False
        for image_ext in self.image_ext:
            image_path = os.path.join(self.data_path, 'JPEGImages', index + image_ext)
            if os.path.exists(image_path):
                image_exist = True
                break
        if not image_exist:
            raise Exception('Image path does not exist: {}'.format(
                os.path.join(self.data_path, 'JPEGImages', index))
            )
        return image_path

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format.
        """
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        boxes, gt_classes = [], []
        for _, obj in enumerate(objs):
            difficult = int(obj.find('difficult').text)
            is_latin = obj.find('language') is None or obj.find('language').text == 'Latin'
            bnd_box = obj.find('bndbox')
            box = [
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymax').text),
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymax').text),
            ]
            label = self.class_to_ind[obj.find('name').text.lower().strip()]
            if difficult:
                continue
            # if self.only_latin and not is_latin:
                # continue
            boxes.append(box)
            gt_classes.append(label)

        return {'boxes': np.array(boxes, dtype=np.int32), 'gt_classes': np.array(gt_classes)}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_path_from_index(self.image_list[i])

    def return_class(self, id):
        id = int(id)
        return self.classes[id]

if __name__ == '__main__':
    pass