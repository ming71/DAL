import cv2
import os
import math
import random
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as transforms

from utils.bbox import rbox_2_quad


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hyp_parse(hyp_path):
    hyp = {}
    keys = [] 
    with open(hyp_path,'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0 : continue
            v = line.strip().split(':')
            try:
                hyp[v[0]] = float(v[1].strip().split(' ')[0])
            except:
                hyp[v[0]] = eval(v[1].strip().split(' ')[0])
            keys.append(v[0])
        f.close()
    return hyp


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))


def curriculum_factor(init, final, step=1, mode='suspend_cosine'):
    if mode == 'cosine':
        sequence = [(0.5 - 0.5 * math.cos(math.pi * i / final)) *  (final - init) + init \
            for i in range(init, final+step, step)]

    elif mode == 'suspend_cosine':
        suspend_ratio = 0.1
        suspend_interval = (final - init)*suspend_ratio
        start = suspend_interval + init if suspend_interval > step else init  
        sequence = [(0.5 - 0.5 * math.cos(math.pi * i / final)) *  (final - init) + init \
             if i>start else init  for i in range(init, final+step, step)]
    # vis 
    import matplotlib.pylab as plt
    import numpy as np
    plt.scatter(np.array([x for x in range(init, final+step, step)]),np.array(sequence)) 
    plt.show()


def plot_gt(img, bboxes, im_path, mode='xyxyxyxy'):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    if mode == 'xywha':
        bboxes = rbox_2_quad(bboxes,mode=mode)
    if mode == 'xyxya':
        bboxes = rbox_2_quad(bboxes,mode=mode)
    for box in bboxes:
        img = cv2.polylines(cv2.UMat(img),[box.reshape(-1,2).astype(np.int32)],True,(0,0,255),2)
        cv2.imwrite(os.path.join('temp','augment_%s' % (os.path.split(im_path)[1])),img)
    print('Check augmentation results in `temp` folder!!!')

if __name__ == '__main__':
    curriculum_factor(836, 6400, 32)


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".bmp", ".png", ".jpg", ".jpeg", ".JPG"])


def rescale(im, target_size, max_size, keep_ratio, multiple=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if keep_ratio:
        # method1
        im_scale = float(target_size) / float(im_size_min)  
        if np.round(im_scale * im_size_max) > max_size:     
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
        # method2
        # im_scale = float(target_size) / float(im_size_max)
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im_scale = np.array([im_scale, im_scale, im_scale, im_scale])

    else:
        target_size = int(np.floor(float(target_size) / multiple) * multiple)
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale


class Rescale(object):
    def __init__(self, target_size=600, max_size=2000, keep_ratio=True):
        self._target_size = target_size
        self._max_size = max_size
        self._keep_ratio = keep_ratio

    def __call__(self, im):
        if isinstance(self._target_size, list):
            random_scale_inds = npr.randint(0, high=len(self._target_size))
            target_size = self._target_size[random_scale_inds]
        else:
            target_size = self._target_size
        im, im_scales = rescale(im, target_size, self._max_size, self._keep_ratio)
        return im, im_scales


class Normailize(object):
    def __init__(self):
        # RGB: https://github.com/pytorch/vision/issues/223
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值和方差
        ])

    def __call__(self, im):
        im = self._transform(im)
        return im


class Reshape(object):
    def __init__(self, unsqueeze=True):
        self._unsqueeze = unsqueeze
        return

    def __call__(self, ims):
        if not torch.is_tensor(ims):
            ims = torch.from_numpy(ims.transpose((2, 0, 1)))
        if self._unsqueeze:
            ims = ims.unsqueeze(0)
        return ims

    
    
 

###
def show_dota_results(img_path,label_path):
    save_path = 'dota_res'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    merged_files = os.listdir(save_path)
    func = get_DOTA_points
    # for folder
    if os.path.isdir(img_path) and os.path.isdir(label_path):
        img_files = os.listdir(img_path)
        xml_files = os.listdir(label_path)
        img_files.sort()	
        xml_files.sort()
        
        img_names = [os.path.splitext(x)[0] for x in img_files]
        xml_names = [os.path.splitext(x)[0] for x in xml_files]
        for img_name in img_names:
            if img_name not in xml_names:
                img_files.remove(img_name+'.png')
#         import ipdb;ipdb.set_trace()
        assert len(img_files) == len(xml_files), 'Not matched between imgs and res!'
        iterations = zip(img_files,xml_files)
        for iter in iterations:
            if iter[0] in merged_files:
                continue
            assert os.path.splitext(iter[0])[0]==os.path.splitext(iter[1])[0],'unmatched images and labels!'   
            # object_coors = get_yolo_points(os.path.join(label_path,iter[1]), rotate=True)
            if not iter[0].endswith('.txt'):
                object_coors = func(os.path.join(label_path,iter[1]),True)
                if len(object_coors):
                    drawbox(os.path.join(img_path,iter[0]),object_coors, save_path =save_path )
                else:
                    print('No obj!')
    
    # for single img
    elif os.path.isfile(label_path):
        object_coors = func(os.path.join(label_path),rotate=False)
        if len(object_coors):
            drawbox(img_path,object_coors,False)
    else:
        print('Path Not Matched!!!')


def drawbox(img_path,object_coors,save_flag=True,save_path=None):
    print(img_path)

    img=cv2.imread(img_path,1)
    for coor in object_coors:
        img = cv2.polylines(img,[coor],True,(0,0,255),2)	
        if save_flag:
            cv2.imwrite(os.path.join(save_path,os.path.split(img_path)[1]), img)
        else: 
            cv2.imshow(img_path,img)
            cv2.moveWindow(img_path,100,100)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



def get_DOTA_points(label_path, rotate=False):
    if not os.path.exists(label_path):
        return []
    with open(label_path,'r') as f:        
        contents=f.read()
        lines=contents.split('\n')
        lines = [x for x in contents.split('\n')  if x]	 

        object_coors=[]	
        for object in lines:
            coors = object.split(' ')
            coors = [int(eval(x)) for x in coors[:-1]]
            x0 = coors[0]; y0 = coors[1]; x1 = coors[2]; y1 = coors[3]
            x2 = coors[4]; y2 = coors[5]; x3 = coors[6]; y3 = coors[7]
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors  