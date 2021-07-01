# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2021/5/8 17:13
# @Software: PyCharm
# @Brief: 不同数据集的父类实现

import numpy as np
import glob
from PIL import Image
import cv2 as cv
import os
import core.config as cfg


class Dataset:
    def __init__(self, target_size=(320, 320), num_classes=21):
        self.target_size = target_size
        self.num_classes = num_classes

    def set_image_info(self, **kwargs):
        """
        Dataset的基类，需要实现self.image_info来存储输入图像的路径和对应mask路径
        :return: None
        """
        pass

    def read_mask(self, image_id):
        """
        读取mask，并转换成语义分割需要的结构。
        :param image_id: 图像的id号
        :return: mask
        """
        mask_path = self.image_info[image_id]["mask_path"]
        image = Image.open(mask_path)
        image = np.array(image)

        image = resize_image_with_pad(image, self.target_size, pad_value=0.0)

        # 转为one hot形式的标签
        h, w = self.target_size
        mask = np.zeros((h, w, self.num_classes), np.uint8)

        for c in range(1, self.num_classes):
            m = np.argwhere(image == c)

            for row, col in m:
                mask[row, col, c] = 1

        return mask


def resize_image_with_pad(image, target_size, pad_value=128.0):
    """
    resize图像，多余的地方用其他颜色填充
    :param image: 输入图像
    :param target_size: resize后图像的大小
    :param pad_value: 填充区域像素值
    :return: image_padded
    """
    image_h, image_w = image.shape[:2]
    input_h, input_w = target_size

    scale = min(input_h / image_h, input_w / image_w)

    image_h = int(image_h * scale)
    image_w = int(image_w * scale)

    dw, dh = (input_w - image_w) // 2, (input_h - image_h) // 2

    if pad_value == 0:
        # mask 用最近领域插值
        image_resize = cv.resize(image, (image_w, image_h), interpolation=cv.INTER_NEAREST)
        image_padded = np.full(shape=[input_h, input_w], fill_value=pad_value)
        image_padded[dh: image_h+dh, dw: image_w+dw] = image_resize
    else:
        # image 用双线性插值
        image_resize = cv.resize(image, (image_w, image_h), interpolation=cv.INTER_LINEAR)
        image_padded = np.full(shape=[input_h, input_w, 3], fill_value=pad_value)
        image_padded[dh: image_h+dh, dw: image_w+dw, :] = image_resize

    return image_padded
