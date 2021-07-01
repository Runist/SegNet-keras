# -*- coding: utf-8 -*-
# @File : losses.py
# @Author: Runist
# @Time : 2021/5/8 17:20
# @Software: PyCharm
# @Brief: 不同loss函数的实现
import tensorflow as tf
from tensorflow.keras import backend, losses


def softmax_sparse_crossentropy(y_true, y_pred):
    """
    当标签的通道数为1时，使用此函数计算loss
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return:
    """
    num_classes = y_pred.shape[-1]

    # reshape成(-1, num_classes)
    y_pred = backend.reshape(y_pred, (-1, num_classes))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = backend.one_hot(tf.cast(backend.flatten(y_true), tf.int32), num_classes)

    loss = -backend.sum(y_true * log_softmax, axis=1)
    loss = backend.mean(loss)

    return loss


def crossentropy_with_logits(y_true, y_pred):
    """
    交叉熵计算loss
    :param y_true:
    :param y_pred:
    :return:
    """
    num_classes = backend.shape(y_pred)[-1]
    # y_true = backend.one_hot(tf.cast(backend.flatten(y_true), tf.int32), num_classes)
    y_true = backend.one_hot(tf.cast(y_true, tf.int32), num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    return backend.mean(loss)


def mse_with_logits(y_true, y_pred):
    """
    当标签的通道数为分类数目时，使用该函数计算loss
    :param y_true:
    :param y_pred:
    :return:
    """
    # num_classes = backend.shape(y_pred)[-1]
    # y_true = backend.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = tf.nn.sigmoid(y_pred)
    y_pred = backend.argmax(y_pred, axis=-1)
    loss = losses.mean_squared_error(y_true, y_pred)

    return backend.mean(loss)


def dice_loss(y_true, y_pred, eps=1e-6):
    """
    dice loss
    :param y_true:
    :param y_pred:
    :param eps: 防止分母为0的系数
    :return: dice loss
    """
    num_classes = backend.shape(y_pred)[-1]

    y_true = backend.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = tf.nn.sigmoid(y_pred)

    # 求得每个sample的每个类的dice
    # intersection = backend.sum(y_true * y_pred, axis=(1, 2, 3))
    # union = backend.sum(y_true, axis=(1, 2, 3)) + backend.sum(y_pred, axis=(1, 2, 3))
    # dices = (2. * intersection + eps) / (union + eps)  # 这个batch下，各个类别的dice
    #
    # dices = backend.mean(dices)                     # 求平均的dice系数
    y_true = backend.flatten(y_true)
    y_pred = backend.flatten(y_pred)

    intersection = backend.sum(y_true * y_pred)
    union = backend.sum(y_true * y_true) + backend.sum(y_pred * y_pred)
    dices = (2. * intersection + eps) / (union + eps)  # 这个batch下，各个类别的dice
    return 1. - dices


def iou_loss(y_true, y_pred, eps=1e-6):
    """
    iou loss
    :param y_true:
    :param y_pred:
    :param eps: 防止分母为0的系数
    :return: iou loss
    """
    num_classes = backend.shape(y_pred)[-1]

    y_pred = tf.nn.sigmoid(y_pred)
    y_true = backend.one_hot(tf.cast(y_true, tf.int32), num_classes)

    y_true = backend.flatten(y_true)
    y_pred = backend.flatten(y_pred)

    intersection = backend.sum(y_true * y_pred)
    union = backend.sum(y_true) + backend.sum(y_pred) - intersection
    loss = (intersection + eps) / (union + eps)

    return 1 - loss


def focal_loss(y_true, y_pred, gamma=2., alpha=.25, eps=1e-6):
    """
    focal loss
    :param y_true:
    :param y_pred:
    :param gamma:
    :param alpha:
    :param eps: 防止log底数为0的系数
    :return:
    """
    num_classes = backend.shape(y_pred)[-1]

    y_pred = tf.nn.sigmoid(y_pred)
    y_true = backend.one_hot(tf.cast(y_true, tf.int32), num_classes)

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    loss = -backend.mean(alpha * backend.pow(1 - pt_1, gamma) * backend.log(pt_1 + eps))\
           -backend.mean((1-alpha) * backend.pow(pt_0, gamma) * backend.log(1 - pt_0 + eps))

    return loss


def seg_loss(y_true, y_pred):
    """
    seg loss 以普通的方式求和
    使用时可以适当缩放至同一数量级
    :param y_true:
    :param y_pred:
    :return:
    """
    loss = crossentropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)\
           + iou_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

    return loss
