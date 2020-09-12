# Dynamic Anchor Learning for Arbitrary-Oriented Object Detection

## Abstract

To be updated.

## Basic Information

The codes build RetinaNet with the proposed DAL method for rotation object detection. The supported datasets include:
* DOTA
* HRSC
* ICDAR2013
* ICDAR2015
* UCAS-AOD
* NWPU VHR-10
* VOC

### Performance

Note that we use only **3** horizontal perset anchors at each location on feature map for rotation detection(while 5 for IC15) . This implementation reaches 24 fps on RTX 2080 Ti.

#### HRSC2016

Note that VOC07 metric is used for evaluation.

| Method          | Backbone   | Input Size | mAP       |
| --------------- | ---------- | ---------- | --------- |
| RetinaNet       | ResNet-50  | 416 x 416  | 80.81     |
| RetinaNet + DAL | ResNet-50  | 416 x 416  | 88.60     |
| RetinaNet + DAL | ResNet-101 | 416 x 416  | 88.95     |
| RetinaNet + DAL | ResNet-101 | 800 x 800  | **89.77** |

#### UCAS-AOD

Refer to this [repo](https://github.com/ming71/UCAS-AOD-benchmark).

#### ICDAR 2015

The performance for long text detection is not good enough. The submissions are shown in the official website: [Incidental Scene Text 2015](https://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1).

| Method              | $P$  | $R$  | $F_1$    |
| ------------------- | ---- | ---- | -------- |
| RetinaNet           | 77.2 | 77.8 | 77.5     |
| RetinaNet + DAL     | 83.7 | 79.5 | 81.5     |
| RetinaNet + DAL(ms) | 84.4 | 80.5 | **82.4** |

#### DOTA

| Method          | Backbone   | mAP       |
| --------------- | ---------- | --------- |
| RetinaNet       | ResNet-50  | 68.43     |
| RetinaNet + DAL | ResNet-50  | 71.44     |
| RetinaNet + DAL | ResNet-101 | 71.78     |
| S2A-Net         | ResNet-50  | 74.12     |
| S2A-Net + DAL   | ResNet-50  | **76.95** |

Experiments on DOTA are implemented based on mmdetection, since the recognition ability of some classes(TC, BC, GTF) is abnormally poor. Trained models are available here.


## Getting Started
### Installation
Build the Cython  and CUDA modules:
```
cd $ROOT/utils
sh make.sh
cd $ROOT/utils/overlaps_cuda
python setup.py build_ext --inplace
```
Installation for DOTA_devkit:
```
cd $ROOT/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
### Inference
You can use the following command to test a dataset. Note that `weight`, `img_dir`, `dataset`,`hyp` should be modified as appropriate.
```
python demo.py
```

### Train
1. Move the dataset to the `$ROOT` directory.
2. Generate imageset files for daatset division via:
```
cd $ROOT/datasets
python generate_imageset.py
```
3. Modify the configuration file `hyp.py` and arguments  in `train.py`, then start training:
```
python train.py
```
### Evaluation
Different datasets use different test methods. For UCAS-AOD/HRSC2016/VOC/NWPU VHR-10, you need to prepare labels in the appropriate format in advance. Take evaluation on HRSC2016 for example:
```
cd $ROOT/datasets/evaluate
python hrsc2gt.py
```
then you can conduct evaluation:
```
python eval.py
```
Note that :

- the script  needs to be executed **only once**, but testing on different datasets needs to be executed again.
- the imageset file used in `hrsc2gt.py` is generated from `generate_imageset.py`.

## Detection Results

![DOTA_results](https://github.com/ming71/DAL/blob/master/DOTA.png)

## Citations

To be updated.