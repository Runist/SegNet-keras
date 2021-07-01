# -*- coding: utf-8 -*-
# @File : MaxUnpoolingWithIndices2D.py
# @Author: Runist
# @Time : 2021/6/23 15:31
# @Software: PyCharm
# @Brief: 反池化层的实现

from tensorflow.keras import layers, backend
import tensorflow as tf


class MaxUnpoolingWithIndices2D(layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpoolingWithIndices2D, self).__init__(**kwargs)
        self.size = size

    @tf.function
    def call(self, inputs, output_shape=None):
        """
        利用index mask来反maxpooling
        :param inputs:
        :param output_shape:
        :return:
        """
        updates, mask = inputs[0], inputs[1]

        mask = backend.cast(mask, 'int32')
        input_shape = tf.shape(updates)

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],      # H
                input_shape[2] * self.size[1],      # W
                input_shape[3],
            )

        one_like_mask = backend.ones_like(mask, dtype='int32')
        batch_shape = backend.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = backend.reshape(
            tf.range(output_shape[0], dtype='int32'), shape=batch_shape
        )

        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        updates_size = tf.size(updates)
        indices = backend.transpose(backend.reshape(backend.stack([b, y, x, f]), [4, updates_size]))
        values = backend.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)

        # ret = tf.where(ret == 0., 1., ret)

        # 获取updates的shape，像上面的操作一样重设shape。
        # 只不过上面是静态图的操作，没有具体的数字
        set_input_shape = updates.get_shape()
        set_output_shape = [
            set_input_shape[0],
            set_input_shape[1] * self.size[0],
            set_input_shape[2] * self.size[1],
            set_input_shape[3]
        ]
        ret.set_shape(set_output_shape)

        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )
