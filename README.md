# DAL

This project hosts the official implementation for our AAAI 2021 paper: 

**Dynamic Anchor Learning for Arbitrary-Oriented Object Detection** [[arxiv](https://arxiv.org/abs/2012.04150)] [[comments](https://zhuanlan.zhihu.com/p/337272217)].

## Abstract

 In this paper, we propose a dynamic anchor learning (DAL) method, which utilizes the newly deﬁned matching degree to comprehensively evaluate the localization potential of the anchors and carry out a more efﬁcient label assignment process. In this way, the detector can dynamically select high-quality anchors to achieve accurate object detection, and the divergence between classiﬁcation and regression will be alleviated. 

## Getting Started

The codes build Rotated RetinaNet with the proposed DAL method for rotation object detection. The supported datasets include: DOTA, HRSC2016, ICDAR2013, ICDAR2015, UCAS-AOD, NWPU VHR-10, VOC. 

### Installation
Insatll requirements:
```
pip install -r requirements.txt
pip install git+git://github.com/lehduong/torch-warmup-lr.git
```
Build the Cython  and CUDA modules:
```
cd $ROOT/utils
sh make.sh
cd $ROOT/utils/overlaps_cuda
python setup.py build_ext --inplace
```
Installation for DOTA_devkit:
```
cd $ROOT/datasets/DOTA_devkit
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

## Main Results


| Method | Dataset     | Bbox | Backbone   | Input Size | mAP/F1 |
| ------ | ----------- | ---- | ---------- | ---------- | ------ |
| DAL    | DOTA        | OBB  | ResNet-101 | 800 x 800  | 71.78  |
| DAL    | UCAS-AOD    | OBB  | ResNet-101 | 800 x 800  | 89.87  |
| DAL    | HRSC2016    | OBB  | ResNet-50  | 416 x 416  | 88.60  |
| DAL    | ICDAR2015   | OBB  | ResNet-101 | 800 x 800  | 82.4   |
| DAL    | ICDAR2013   | HBB  | ResNet-101 | 800 x 800  | 81.3   |
| DAL    | NWPU VHR-10 | HBB  | ResNet-101 | 800 x 800  | 88.3   |
| DAL    | VOC 2007    | HBB  | ResNet-101 | 800 x 800  | 76.1   |


## Detections

![DOTA_results](https://github.com/ming71/DAL/blob/master/outputs/DOTA.png)

## Citation

If you find our work or code useful in your research, please consider citing:

```
@inproceedings{ming2021dynamic,
  title={Dynamic Anchor Learning for Arbitrary-Oriented Object Detection},
  author={Ming, Qi and Zhou, Zhiqiang and Miao, Lingjuan and Zhang, Hongwei and Li, Linhao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2355--2363},
  year={2021}
}
```

If you have any questions, please contact me via issue or [email](mq_chaser@126.com).