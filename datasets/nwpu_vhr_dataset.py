import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from utils.augment import *
from utils.utils import plot_gt
from utils.bbox import quad_2_rbox, constraint_theta

class NWPUDataset(data.Dataset):

    def __init__(self,
                 dataset= None, 
                 augment = False,
                 level = 1,
                 ):
        self.image_set_path = dataset
        if dataset is not None:
            self.image_list = self._load_image_names()  
        self.classes = ('__background__', 'airplane', 'ship', 'storage-tank', 
                        'baseball-diamond', 'tennis-court', 'basketball-court', 
                        'ground-track-field', 'harbor', 'bridge', 'vehicle')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))    
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_path = self.image_list[index]  
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roidb = self._load_annotation(self.image_list[index])
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
        bboxes = roidb['boxes'][gt_inds, :]
        classes = roidb['gt_classes'][gt_inds]
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        if self.augment :
            transform = Augment([ 
                                HorizontalFlip(0.5),
                                VerticalFlip(0.5),
                                Affine(degree=0, translate=0.1, scale=0., p=0.5),
                                ],box_mode = 'xyxyxyxy',)
            im, bboxes = transform(im, bboxes)

        mask = mask_valid_boxes(quad_2_rbox(bboxes,'xywha'), return_mask=True)
        bboxes = bboxes[mask]
        gt_boxes = gt_boxes[mask]
        classes = classes[mask]

        for i, bbox in enumerate(bboxes):
            gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode = 'xyxya')   # 四点转xyxya（a为角度制）
            gt_boxes[i, 5] = classes[i]

        ## test augmentation
        # print(im.shape)
        # plot_gt(im, gt_boxes[:,:5], im_path, mode = 'xyxya')

        return {'image': im, 'boxes': gt_boxes, 'path': im_path}

    def _load_image_names(self):
        """
        Load the names listed in this dataset's image set file.
        """
        image_set_file = self.image_set_path
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]
        return image_list


    def _load_annotation(self, index):
        root_dir, img_name = os.path.split(index)
        filename = os.path.join(root_dir.replace('AllImages','Annotations'), img_name[:-4]+'.txt')
        
        boxes, gt_classes = [], []
        with open(filename,'r',encoding='utf-8-sig') as f:
            content = f.read().strip()
            objects = content.split('\n')
            for obj in objects:
                if len(obj) != 0 :
                    line = obj.replace('(', '').replace(')', '').strip().split(',')
                    assert len(line)==5, 'Wronging occurred in load_annos'
                    label = eval(line[-1])
                    box = [ eval(x) for x in  line[:4] ]
                    points = [box[0], box[1], box[0], box[3],
                              box[2], box[3], box[2], box[1]]
                    boxes.append(points)
                    gt_classes.append(label)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}


    def return_class(self, id):
        id = int(id)
        return self.classes[id]

            


if __name__ == '__main__':
    pass
