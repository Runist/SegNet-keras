# -*- coding: utf-8 -*-
# @File : BilinearUpSampling.py
# @Author: Runist
# @Time : 2021/5/8 16:19
# @Software: PyCharm
# @Brief:

from tensorflow.keras import layers, backend
import tensorflow as tf
import numpy as np


def resize_images_bilinear(inputs, height_factor=1, width_factor=1, target_height=None, target_width=None):
    """
    调整形状的4D张量中包含的图像的大小, 乘以（height_factor，width_factor）。两个因子都应为正整数。
    :param inputs:
    :param height_factor:
    :param width_factor:
    :param target_height:
    :param target_width:
    :return:
    """

    original_shape = inputs.shape

    if target_height and target_width:
        new_shape = np.array((target_height, target_width))
    else:
        new_shape = inputs.shape[1:3]
        new_shape *= np.array([height_factor, width_factor])

    inputs = tf.image.resize(inputs, new_shape)

    if target_height and target_width:
        inputs.set_shape((None, target_height, target_width, None))
    else:
        inputs.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))

    return inputs


class BilinearUpSampling2D(layers.Layer):
    def __init__(self, scale=(1, 1), target_size=None, **kwargs):

        self.scale = tuple(scale)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None

        self.input_spec = [layers.InputSpec(ndim=4)]

        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        width = int(self.scale[0] * input_shape[1] if input_shape[1] is not None else None)
        height = int(self.scale[1] * input_shape[2] if input_shape[2] is not None else None)
        if self.target_size is not None:
            width = self.target_size[0]
            height = self.target_size[1]

        return input_shape[0], width, height, input_shape[3]

    def call(self, x, mask=None):
        if self.target_size:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1])
        else:
            return resize_images_bilinear(x, height_factor=self.scale[0], width_factor=self.scale[1])

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
