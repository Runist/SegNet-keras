# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/5/11 9:12
# @Software: PyCharm
# @Brief: 训练脚本
from tensorflow.keras import optimizers, callbacks, utils
from core.VOCdataset import VOCDataset
from nets.SegNet import *
from core.losses import *
from core.metrics import *
from core.callback import *
import core.config as cfg
import tensorflow as tf
import os


def train_by_fit(model, epochs, train_gen, test_gen, train_steps, test_steps):
    """
    fit方式训练
    :param model: 训练模型
    :param epochs: 训练轮数
    :param train_gen: 训练集生成器
    :param test_gen: 测试集生成器
    :param train_steps: 训练次数
    :param test_steps: 测试次数
    :return: None
    """

    cbk = [
        callbacks.ModelCheckpoint(
            './weights/epoch={epoch:02d}_val_loss={val_loss:.04f}_miou={val_object_miou:.04f}.h5',
            save_weights_only=True),
    ]

    learning_rate = CosineAnnealingLRScheduler(2*epochs, train_steps, 1e-4, 1e-6, warmth_rate=0.05)
    optimizer = optimizers.Adam(learning_rate)
    lr_info = print_lr(optimizer)

    model.compile(optimizer=optimizer,
                  loss=crossentropy_with_logits,
                  metrics=[object_accuracy, object_miou, lr_info])

    # trainable_layer = 92
    trainable_layer = 19
    for i in range(trainable_layer):
        print(model.layers[i].name)
        model.layers[i].trainable = False
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=epochs,
              callbacks=cbk)

    # learning_rate = CosineAnnealingLRScheduler(epochs, train_steps, 1e-5, 1e-6, warmth_rate=0.1)
    # optimizer = optimizers.Adam(learning_rate)
    # lr_info = print_lr(optimizer)
    #
    # model.compile(optimizer=optimizer,
    #               loss=crossentropy_with_logits,
    #               metrics=[object_accuracy, object_miou, lr_info])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    print('train all layers.')

    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=epochs * 2,
              initial_epoch=epochs,
              callbacks=cbk)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists("weights"):
        os.mkdir("weights")

    # model = SegNet(cfg.input_shape, cfg.num_classes)
    # model.load_weights("segnet_model.h5", skip_mismatch=True, by_name=True)
    model = SegNet_VGG16(cfg.input_shape, cfg.num_classes)
    model.summary()

    train_dataset = VOCDataset(cfg.train_txt_path, batch_size=cfg.batch_size, aug=True)
    test_dataset = VOCDataset(cfg.val_txt_path, batch_size=cfg.batch_size)

    train_steps = len(train_dataset) // cfg.batch_size
    test_steps = len(test_dataset) // cfg.batch_size

    train_gen = train_dataset.tf_dataset()
    test_gen = test_dataset.tf_dataset()

    train_by_fit(model, cfg.epochs, train_gen, test_gen, train_steps, test_steps)
