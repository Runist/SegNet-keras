# -*- coding: utf-8 -*-
# @File : evaluate.py
# @Author: Runist
# @Time : 2021/5/19 12:13
# @Software: PyCharm
# @Brief: 测试miou脚本
import numpy as np
import os
import core.config as cfg
from core.VOCdataset import VOCDataset
from core.metrics import get_confusion_matrix_and_miou
from nets.SegNet import *
from tqdm import tqdm
import tensorflow as tf


def evaluate(model, val_file_path, num_classes):
    """
    评价SegNet网络指标，主要是测试miou
    :param model: 模型对象
    :param val_file_path: 验证集文件路径
    :param num_classes: 分类数量
    :return: None
    """
    val_dataset = VOCDataset(val_file_path, batch_size=1)
    val_dataset = val_dataset.tf_dataset()
    val_dataset = iter(val_dataset)

    f = open(val_file_path, mode='r')
    images = f.readlines()
    num_sample = len(images)

    sum_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    process_bar = tqdm(range(num_sample), ncols=100, unit="step")

    for i in process_bar:
        image, y_true = next(val_dataset)

        y_pred = model.predict(image)
        y_pred = tf.nn.softmax(y_pred)

        y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)
        y_pred = np.squeeze(y_pred, axis=0)
        y_true = np.squeeze(y_true, axis=0).astype(np.uint8)

        confusion_matrix, miou = get_confusion_matrix_and_miou(y_true, y_pred, num_classes=21)
        sum_confusion_matrix += confusion_matrix

        process_bar.set_postfix(image_id=images[i].strip(), miou="{:.4f}".format(miou))

    intersection = np.diag(sum_confusion_matrix)
    union = np.sum(sum_confusion_matrix, axis=0) + np.sum(sum_confusion_matrix, axis=1) - intersection

    iou = intersection / union
    iou = np.nan_to_num(iou)    # 避免计算iou时出现nan

    meanIOU = np.mean(iou)
    object_meanIOU = np.mean(iou[1:])
    print("-"*80)
    print("Total MIOU: {:.4f}".format(meanIOU))
    print("Object MIOU: {:.4f}".format(object_meanIOU))
    print('pixel acc: {:.4f}'.format(np.sum(intersection)/np.sum(sum_confusion_matrix)))
    print("IOU: ", iou)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # model = SegNet(cfg.input_shape, cfg.num_classes)
    model = SegNet_VGG16(cfg.input_shape, cfg.num_classes)
    model.load_weights("./weights/segnet_weights.h5")

    evaluate(model, cfg.val_txt_path, cfg.num_classes)
