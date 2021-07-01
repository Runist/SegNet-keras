# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2021/5/11 11:00
# @Software: PyCharm
# @Brief: 配置文件

input_shape = (320, 320, 3)
num_classes = 21

epochs = 30
batch_size = 4
lr = 0.0001

train_txt_path = "D:/Code/Data/VOC2012/ImageSets/SegmentationAug/train.txt"
val_txt_path = "D:/Code/Data/VOC2012/ImageSets/SegmentationAug/val.txt"
trainval_txt_path = "D:/Code/Data/VOC2012/ImageSets/SegmentationAug/trainval.txt"
