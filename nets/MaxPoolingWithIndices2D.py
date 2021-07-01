# -*- coding: utf-8 -*-
# @File : MaxPoolingWithIndices2D.py
# @Author: Runist
# @Time : 2021/6/23 15:24
# @Software: PyCharm
# @Brief: 带索引的MaxPooling layer

from tensorflow.keras import layers, backend
import tensorflow as tf


class MaxPoolingWithIndices2D(layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithIndices2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    @tf.function
    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if backend.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                backend.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = backend.cast(argmax, backend.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        """
        计算输出shape
        :param input_shape: model输出shape
        :return: 两个shape
        """
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)

        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


if __name__ == '__main__':
    import numpy as np

    arr = np.array([[[0, 3, 7, 5],
                     [1, 5, 4, 6],
                     [4, 3, 1, 5],
                     [1, 3, 4, 6]],
                    [[0, 3, 7, 5],
                     [1, 5, 4, 6],
                     [4, 3, 1, 5],
                     [1, 3, 4, 6]]])

    # arr = np.expand_dims(arr, axis=0)
    arr = np.expand_dims(arr, axis=-1)

    updates, argx = MaxPoolingWithIndices2D()(arr)
    print(argx.shape)
    updates, mask = updates, argx
    mask = backend.cast(mask, 'int32')
    input_shape = updates.shape

    output_shape = (
        input_shape[0],
        input_shape[1] * 2,      # H
        input_shape[2] * 2,      # W
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


    print(ret)