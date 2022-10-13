# Keras  - SegNet

## Part 1. Introduction

Segnet is a deep network of image semantic segmentation proposed by Cambridge to solve the problem of automatic driving or intelligent robot. It refers to FCN and uses the indexed maxpooling layer as the down sample, while taking the index as the parameter to the maxunpooling layer as the up sample.

![img2.jpg](https://i.loli.net/2021/07/01/vkmZc3xz4SdsgAX.jpg)

### Models for this repository

The SegNet paper introduces a encoder similar to VGG16. But with an additional BatchNormalization layer after each convolution layer compared to VGG16. I couldn't load weights directly from Keras team. Therefore, I delete the BatchNormalization layer to match the VGG16.

| Model name   | Dataset                | MIoU   | Pixel accuracy |
| ------------ | ---------------------- | ------ | -------------- |
| SegNet_VGG16 | VOC train dataset      | 0.9077 | 0.9817         |
|              | VOC validation dataset | 0.5708 | 0.9224         |



## Part 2. Quick  Start

1. Pull this repository.

```shell
git clone https://github.com/Runist/SegNet-keras.git
```

2. You need to install some dependency package.

```shell
cd SegNet-keras
pip install -r requirements.txt
```

3. Download the *[VOC](https://www.kaggle.com/huanghanchina/pascal-voc-2012)* dataset(VOC [SegmetationClassAug](http://home.bharathh.info/pubs/codes/SBD/download.html) if you need).  
4. Getting SegNet weights.

```shell
wget https://github.com/Runist/SegNet-keras/releases/download/v0.1/segnet_weights.h5
```

5. Run **predict.py**, you'll see the result of SegNet.

```shell
python predict.py
```

Input image:

![2007_000129.jpg](https://i.loli.net/2021/06/30/wetEJVlFqZ9digL.jpg)

Output image（resize to 320 x 320）:

![segnet.jpg](https://i.loli.net/2021/07/01/rhszImEFviVktWJ.jpg)

## Part 3. Train your own dataset

1. You should rewrite your data pipeline, *Dateset* where in *dataset.py* is the base class, such as  *VOCdataset.py*. 

```python
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
```

2. Start training.

```shell
python train.py
```

3. Running *evaluate.py* to get mean iou and pixel accuracy.

```shell
python evaluate.py
--------------------------------------------------------------------------------
Total MIOU: 0.5708
Object MIOU: 0.5531
pixel acc: 0.9224
IOU:  [0.92533772 0.7417258  0.56996184 0.56150997 0.40248062 0.46306653
 0.81379056 0.72784183 0.76995939 0.27809084 0.53515977 0.30562614
 0.65339    0.6333953  0.6229686  0.7550784  0.27868099 0.44803004
 0.35444195 0.61200612 0.53468034]
```

   

## Part 4. Reference and other implement

- Paper: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)

- Paper with the code: [alexgkendall](https://github.com/alexgkendall)/[caffe-segnet](https://github.com/alexgkendall/caffe-segnet)

- [chandnii7](https://github.com/chandnii7)/[Image-Segmentation](https://github.com/chandnii7/Image-Segmentation)

  
