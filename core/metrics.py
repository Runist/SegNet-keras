# -*- coding: utf-8 -*-
# @File : metrics.py
# @Author: Runist
# @Time : 2021/5/12 16:12
# @Software: PyCharm
# @Brief: 训练中的评价指标
import core.config as cfg
from tensorflow.keras import backend, metrics, callbacks
import tensorflow as tf
import numpy as np


def get_confusion_matrix(y_true, y_pred, num_classes=cfg.num_classes):
    """
    计算混淆矩阵
    :param y_true: 标签
    :param y_pred: 预测
    :param num_classes: 分类数量
    :return: confusion_matrix
    """
    y_pred = backend.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # 避免mask中有未标注的像素
    mask = (y_true >= 0) & (y_true < num_classes)
    label = num_classes * y_true[mask] + y_pred[mask]
    label = tf.cast(label, tf.int32)

    # 利用bincount计算混淆矩阵
    count = tf.math.bincount(label, minlength=num_classes**2)  # 核心代码
    confusion_matrix = backend.reshape(count, (num_classes, num_classes))

    return confusion_matrix


def object_accuracy(y_true, y_pred, num_classes=cfg.num_classes):
    """
    以像素点计算前景的准确率
    :param y_true: 标签
    :param y_pred: 预测
    :param num_classes: 分类数量
    :return: acc
    """
    confusion_matrix = get_confusion_matrix(y_true, y_pred, num_classes)
    intersection = tf.linalg.diag_part(confusion_matrix)[1:]

    acc = tf.reduce_sum(intersection) / tf.reduce_sum(confusion_matrix)
    acc = tf.where(tf.math.is_nan(acc), tf.zeros_like(acc), acc)

    return acc


def object_miou(y_true, y_pred, num_classes=cfg.num_classes):
    """
    衡量图中目标的iou
    :param y_true: 标签
    :param y_pred: 预测
    :param num_classes: 分类数量
    :return: miou
    """
    confusion_matrix = get_confusion_matrix(y_true, y_pred, num_classes)

    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)

    # 取对角元素的值，对角线上的值可认为是TP或是交集
    intersection = tf.linalg.diag_part(confusion_matrix)
    # axis = 1表示混淆矩阵行的值；axis = 0表示取混淆矩阵列的值，都是返回一个一维列表，需要求和
    union = tf.reduce_sum(confusion_matrix, axis=1) + tf.reduce_sum(confusion_matrix, axis=0) - intersection

    intersection = intersection
    union = union

    iou = intersection / union  # 其值为各个类别的IoU
    # 避免nan
    iou = tf.where(tf.math.is_nan(iou), tf.zeros_like(iou), iou)
    # 不求包含背景部分的iou
    miou = tf.reduce_mean(iou[1:])

    return miou


def get_confusion_matrix_and_miou(y_true, y_pred, num_classes):
    """
    计算miou和混淆矩阵
    :param y_true:
    :param y_pred:
    :param num_classes:
    :return:
    """
    # 避免mask中有未标注的像素
    mask = (y_true >= 0) & (y_true < num_classes)
    label = num_classes * y_true[mask].astype(int) + y_pred[mask]

    # 利用bincount计算混淆矩阵
    count = np.bincount(label, minlength=num_classes**2)  # 核心代码
    confusion_matrix = count.reshape(num_classes, num_classes)

    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)

    # 取对角元素的值，对角线上的值可认为是TP或是交集
    intersection = np.diag(confusion_matrix)
    # axis = 1表示混淆矩阵行的值； axis = 0表示取混淆矩阵列的值，都是返回一个一维列表，需要求和
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - intersection

    # 不求包含背景部分的iou
    intersection = intersection
    union = union

    iou = intersection / union  # 其值为各个类别的IoU
    iou = np.nan_to_num(iou[1:])    # 避免计算iou时出现nan
    miou = np.mean(iou)         # 求各类别IoU的平均

    return confusion_matrix, miou

