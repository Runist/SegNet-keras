# -*- coding: utf-8 -*-
# @File : VOCdataset.py
# @Author: Runist
# @Time : 2021/5/10 12:17
# @Software: PyCharm
# @Brief: voc数据集的读取脚本
from core.dataset import Dataset
import numpy as np
import random
import tensorflow as tf
from PIL import Image, ImageEnhance
import os
import cv2 as cv


class VOCDataset(Dataset):
    def __init__(self, annotation_path, batch_size=4, target_size=(320, 320), num_classes=21, aug=False):
        super().__init__(target_size, num_classes)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.annotation_path = annotation_path
        self.aug = aug
        self.read_voc_annotation()
        self.set_image_info()

    def __len__(self):
        return len(self.annotation)

    def read_voc_annotation(self):
        self.annotation = []
        f = open(self.annotation_path, mode='r')
        for image_id in f.readlines():
            self.annotation.append(image_id.strip())

    def set_image_info(self):
        """
        继承自Dataset类，需要实现对输入图像路径的读取和mask路径的读取，且存储到self.image_info中
        :return:
        """
        self.image_info = []
        root = os.path.split(self.annotation_path)[0]
        root = os.path.split(root)[0]
        root = os.path.split(root)[0]
        # random.shuffle(self.annotation)

        for image_id in self.annotation:
            image_path = "{}/JPEGImages/{}".format(root, image_id + '.jpg')
            mask_path = "{}/SegmentationClassAug/{}".format(root, image_id + '.png')
            self.image_info.append({"image_path": image_path, "mask_path": mask_path})

    def read_mask(self, image_id, one_hot=False):
        """
        读取mask，并转换成语义分割需要的结构。
        :param image_id: 图像的id号
        :param one_hot: 是否转成one hot的形式
        :return: image
        """
        mask_path = self.image_info[image_id]["mask_path"]
        image = Image.open(mask_path)
        image = np.array(image)

        image[image == 255] = 0

        if one_hot:
            # 转为one hot形式的标签
            h, w = image.shape[:2]
            mask = np.zeros((h, w, self.num_classes), np.uint8)

            for c in range(1, self.num_classes):
                m = np.argwhere(image == c)

                for row, col in m:
                    mask[row, col, c] = 1

            return mask

        return image

    def read_image(self, image_id):
        """
        读取图像
        :return: image
        """
        image_path = self.image_info[image_id]["image_path"]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        return image

    def resize_image_with_pad(self, image, target_size, pad_value=128.0):
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

    def random_horizontal_flip(self, image, mask):
        """
        左右翻转图像
        :param image: 输入图像
        :param mask: 对应的mask图
        :return:
        """
        _, w, _ = image.shape
        image = cv.flip(image, 1)
        mask = cv.flip(mask, 1)

        return image, mask

    def random_crop(self, image, mask):
        """
        随机裁剪
        :param image: 输入图像
        :param mask: 对应的mask图
        :return:
        """
        h, w, _ = image.shape

        max_l_trans = w // 10
        max_u_trans = h // 10
        max_r_trans = w - w // 10
        max_d_trans = h - h // 10

        crop_xmin = int(random.uniform(0, max_l_trans))
        crop_ymin = int(random.uniform(0, max_u_trans))
        crop_xmax = int(random.uniform(max_r_trans, w))
        crop_ymax = int(random.uniform(max_d_trans, h))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
        mask = mask[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        return image, mask

    def random_translate(self, image, mask):
        """
        整图随机位移
        :param image: 输入图像
        :param mask: 对应的mask图
        :return:
        """

        h, w, _ = image.shape

        max_l_trans = h // 10
        max_u_trans = w // 10
        max_r_trans = h // 10
        max_d_trans = w // 10

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])

        image = cv.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))
        mask = cv.warpAffine(mask, M, (w, h), borderValue=(0, 0, 0))

        return image, mask

    def color_jitter(self, image, hue=0.1, sat=1.5, val=1.5):
        """
        色域抖动数据增强
        :param image: 输入图像
        :param hue: 色调
        :param sat: 饱和度
        :param val: 明度
        :return: image
        """
        image = np.array(image, np.float32) / 255
        image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

        h = random.uniform(-hue, hue)
        s = random.uniform(1, sat) if random.random() < .5 else 1/random.uniform(1, sat)
        v = random.uniform(1, val) if random.random() < .5 else 1/random.uniform(1, val)

        image[..., 0] += h * 360
        image[..., 0][image[..., 0] > 1] -= 1.
        image[..., 0][image[..., 0] < 0] += 1.
        image[..., 1] *= s
        image[..., 2] *= v
        image[image[:, :, 0] > 360, 0] = 360
        image[:, :, 1:][image[:, :, 1:] > 1] = 1
        image[image < 0] = 0

        image = cv.cvtColor(image, cv.COLOR_HSV2RGB) * 255
        image = image.astype(np.uint8)

        return image

    def random_brightness(self, image, brightness_range):
        """
        随机亮度加减
        :param image: 输入图像
        :param brightness_range: 亮度加减范围
        :return: image
        """
        image = np.array(image, np.float32) / 255
        hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        h, s, v = cv.split(hsv)

        value = random.uniform(-brightness_range, brightness_range)

        v += value
        v[v > 1] = 1.
        v[v < 0] = 0.

        final_hsv = cv.merge((h, s, v))
        image = cv.cvtColor(final_hsv, cv.COLOR_HSV2RGB) * 255
        image = image.astype(np.uint8)

        return image

    def random_sharpness(self, image, sharp_range=3.):
        """
        随机锐度加强
        :param image: 输入图像
        :param sharp_range: 锐度加减范围
        :return: image
        """
        image = Image.fromarray(image)
        enh_sha = ImageEnhance.Sharpness(image)
        image = enh_sha.enhance(random.uniform(-0.5, sharp_range))
        image = np.array(image)

        return image

    def parse(self, index):
        """
        tf.data的解析器
        :param index: 字典索引
        :return:
        """

        def get_data(i):
            image = self.read_image(i)
            mask = self.read_mask(i, one_hot=False)

            if random.random() < 0.4 and self.aug:
                if random.random() < 0.5 and self.aug:
                    image, mask = self.random_horizontal_flip(image, mask)
                if random.random() < 0.5 and self.aug:
                    image = self.color_jitter(image)
                if random.random() < 0.5 and self.aug:
                    image = self.random_brightness(image, brightness_range=0.3)
                if random.random() < 0.5 and self.aug:
                    image = self.random_sharpness(image, sharp_range=3.)
                if random.random() < 0.5 and self.aug:
                    image, mask = self.random_crop(image, mask)
                if random.random() < 0.5 and self.aug:
                    image, mask = self.random_translate(image, mask)

            image = self.resize_image_with_pad(image, self.target_size)
            mask = self.resize_image_with_pad(mask, self.target_size, pad_value=0.)
            # image = image / 255
            image -= [123.68, 116.68, 103.94]

            return image, mask

        image, mask = tf.py_function(get_data, [index], [tf.float32, tf.float32])
        h, w = self.target_size
        image.set_shape([h, w, 3])
        mask.set_shape([h, w])

        return image, mask

    def tf_dataset(self):
        """
        用tf.data的方式读取数据，以提高gpu使用率
        :return: 数据集
        """
        index = [i for i in range(len(self))]
        # 这是GPU读取方式
        dataset = tf.data.Dataset.from_tensor_slices(index)

        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
