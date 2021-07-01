# -*- coding: utf-8 -*-
# @File : SegNet.py
# @Author: Runist
# @Time : 2021/6/15 17:38
# @Software: PyCharm
# @Brief: SegNet的实现

from tensorflow.keras import layers, models
from tensorflow.python.keras.utils import data_utils
from nets.MaxPoolingWithIndices2D import MaxPoolingWithIndices2D
from nets.MaxUnpoolingWithIndices2D import MaxUnpoolingWithIndices2D
import os


def SegNet(input_shape, num_classes):
    """
    论文中介绍的SegNet网络
    :param input_shape: 模型输入shape
    :param num_classes: 分类数量
    :return: model
    """
    inputs = layers.Input(shape=input_shape)

    # encoder
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_1 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_2 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_3 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_4 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_5 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # decoder
    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_5])
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_4])
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_3])
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_2])
    x = layers.Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_1])
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_classes, (1, 1), padding='valid', kernel_initializer='he_uniform')(x)
    outputs = layers.BatchNormalization()(x)

    # outputs = layers.Activation('softmax')(x)

    segnet_model = models.Model(inputs=inputs, outputs=outputs, name='SegNet')

    return segnet_model


def SegNet_VGG16(input_shape, num_classes):
    """
    VGG16作为encoder实现的SegNet
    :param input_shape: 模型输入shape
    :param num_classes: 分类数量
    :return: model
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x, mask_1 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x, mask_2 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x, mask_3 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x, mask_4 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x, mask_5 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    # decoder
    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_5])
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_4])
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_3])
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_2])
    x = layers.Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_1])
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Conv2D(num_classes, (1, 1), padding='valid', kernel_initializer='he_uniform')(x)

    model = models.Model(inputs, outputs)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    weights_path = data_utils.get_file(
        weights_path,
        'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model
