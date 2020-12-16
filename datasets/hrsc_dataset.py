import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from utils.augment import *
from utils.utils import plot_gt
from utils.bbox import quad_2_rbox, constraint_theta

class HRSCDataset(data.Dataset):

    def __init__(self,
                 dataset= None,  
                 augment = False,
                 level = 1,
                 ):
        self.image_set_path = dataset
        if dataset is not None:
            self.image_list = self._load_image_names()  
        self.level = level
        if self.level == 1:
            self.classes = ('__background__', 'ship') 
        if self.level == 2:
            self.classes = ('__background__', 'ship', 'air.', 'war.','mer.') 
        if self.level == 3:
            self.classes = ('__background__', 'ship' , 'air.', 'war.','mer.', 'Nim.', 
                            'Ent.' , 'Arl.' , 'Whi.' , 'Per.' , 'San.' , 'Tic.' , 'Aus.' , 
                            'Tar.' , 'Con.' , 'Com.A' , 'Car.A' , 'Con.A' , 'Med.' , 'Car.B' )     
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
        nt = len(roidb['boxes'])
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        if nt:
            bboxes = roidb['boxes'][gt_inds, :]
            classes = roidb['gt_classes'][gt_inds]
            if self.augment :
                transform = Augment([ HSV(0.5, 0.5, p=0.5),
                                    HorizontalFlip(0.5),
                                    VerticalFlip(0.5),
                                    Affine(degree=10, translate=0.1, scale=0.1, p=0.5),
                                    ],box_mode = 'xywha',)
                im, bboxes = transform(im, bboxes)
            gt_boxes[:, :-1] = bboxes

            mask = mask_valid_boxes(bboxes, return_mask=True)
            bboxes = bboxes[mask]
            gt_boxes = gt_boxes[mask]
            classes = classes[mask]

            for i, bbox in enumerate(bboxes):
                gt_boxes[:, 5] = classes[i]
            gt_boxes = constraint_theta(gt_boxes)
            cx, cy, w, h = [gt_boxes[:, x] for x in range(4)]      
            x1 = cx - 0.5*w
            x2 = cx + 0.5*w
            y1 = cy - 0.5*h
            y2 = cy + 0.5*h
            gt_boxes[:,0] = x1;  gt_boxes[:,1] = y1; gt_boxes[:,2] = x2; gt_boxes[:,3] = y2

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
        filename = os.path.join(root_dir.replace('AllImages','Annotations'), img_name[:-4]+'.xml')
        boxes, gt_classes = [], []
        with open(filename,'r',encoding='utf-8-sig') as f:
            content = f.read()
            assert '<HRSC_Object>'  in content, 'Background picture occurred in %s'%filename
            objects = content.split('<HRSC_Object>')
            info = objects.pop(0)
            for obj in objects:
                assert len(obj) != 0, 'No onject found in %s'%filename
                cls_id = obj[obj.find('<Class_ID>')+10 : obj.find('</Class_ID>')]
                diffculty = obj[obj.find('<difficult>')+11 : obj.find('</difficult>')]
                if cls_id in ['100000027', '100000022'] or diffculty == '1':
                    continue
                cx = round(eval(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')]))
                cy = round(eval(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')]))
                w  = round(eval(obj[obj.find('<mbox_w>')+8 : obj.find('</mbox_w>')]))
                h  = round(eval(obj[obj.find('<mbox_h>')+8 : obj.find('</mbox_h>')]))
                a  = eval(obj[obj.find('<mbox_ang>')+10 : obj.find('</mbox_ang>')])/math.pi*180
                box = np.array([cx, cy, w, h, a])
                boxes.append(box)
                label = self.class_mapping(cls_id, self.level)
                gt_classes.append(label)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}

    def class_mapping(self, cls_id, level):
        if level == 1:
            return 1
        if level == 2:
            if cls_id in ['100000005','100000006','100000012','100000013','100000016','10000032']:
                cls_id =  '100000002'
            if cls_id in ['100000007','100000008','100000009','100000010','100000011','10000015',
                          '10000017','10000019','10000028','10000029']:
                cls_id =  '100000003'
            if cls_id in ['100000018','100000020','100000024','100000025','100000026','10000030']:
                cls_id = '100000004'
            class_ID = ['bg', '100000001', '100000002', '100000003', '100000004']
            return class_ID.index(cls_id)
        if level == 3:
            if cls_id in ['1000000012', '1000000013', '1000000032',]:
                cls_id = '100000002'
            if cls_id in ['100000017', '100000028']:
                cls_id = '100000003'
            if cls_id in ['100000024', '100000026']:
                cls_id = '100000004'
            class_ID = ['bg', '100000001', '100000002', '100000003', '100000004', '100000005', 
                        '100000006', '100000007', '100000008', '100000009', '1000000010',
                        '1000000011', '100000015','1000000016', '100000018', '100000019', 
                        '100000020', '100000025' , '100000029', '100000030'] 
            return class_ID.index(cls_id)

    def return_class(self, id):
        id = int(id)
        return self.classes[id]

            


if __name__ == '__main__':
    pass
