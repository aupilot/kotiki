import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(1337)  # for reproducibility

from collections import namedtuple
import pandas as pd
import math
import os
import re
import sealiondata
import random

import matplotlib.pyplot as plt
from scipy.misc import imresize
from multiprocessing.pool import ThreadPool

import img_augmentation
import utils

LionPos = namedtuple('LionPos', ['train_id', 'cls', 'row', 'col'])
scale_dir = "../Sealion/TrainScale/"

crop_size = 224
input_shape = (crop_size, crop_size, 3)

scale_categories = 32
crops_count = 513

class SampleCfg:
    """
    Configuration structure for crop parameters.

    Crop from img[row:row+crop_size/scale, col:col+crop_size/scale] 
    """

    def __init__(self, img, train_idx, row, col,
                 scale=1.0,  # 1.0 - no changes, 0.5 - zoomed out, scale 2x bigger area to crop
                 saturation=0.5, contrast=0.5, brightness=0.5,  # 0.5  - no changes, range 0..1
                 hflip=False,
                 vflip=False):
        self.img = img
        self.train_idx = train_idx
        self.row = row
        self.col = col
        self.scale = scale
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __lt__(self, other):
        return True

class ScaleDataset:
    def __init__(self, core_dataset, preprocess_input, use_categorical=False, scale_dataset_dir='../Sealion/TrainScale/'):
        self.core_dataset = core_dataset
        self.all_items = []
        self.crop_size = input_shape[0]
        self.preprocess_input = preprocess_input
        self.use_categorical = use_categorical
        self.items_scale = {}

        for i in range(1000):
            fname = '{}/{}_scale.txt'.format(scale_dataset_dir, i)
            try:
                f = open(fname, 'r')
                scale = float(f.read())
                self.all_items.append((i, scale))
                self.items_scale[i] = scale
            except IOError:
                pass

        self.train_items, self.test_items, _, _ = train_test_split(self.all_items, [0] * len(self.all_items),
                                                                   train_size=0.8,
                                                                   random_state=42)

    def load_img(self, train_id):
        return self.core_dataset.load_train_image(train_id=train_id, border=0, mask=True)

    def gen_img_crop(self, img, row, col):
        return utils.crop_zero_pad(img, x=col, y=row, w=self.crop_size, h=self.crop_size)
        # return img[row:row + self.img_size, col:col + self.img_size]

    def scale_to_one_shot(self, scale):
        idx = np.clip((math.log(scale, 2) + 1) * scale_categories / 2, 0, scale_categories - 1)
        res = np.zeros((scale_categories,))
        res[int(idx)] = 1.0
        if 0.5 < idx < scale_categories - 1.5:
            res[int(idx - 1)] = 0.5
        if 1.5 < idx < scale_categories - 1.5:
            res[int(idx + 1)] = 0.5
        return res

    def gen_x(self, cfg: SampleCfg):
        if cfg.scale == 1.0:
            crop = self.gen_img_crop(img=cfg.img, row=cfg.row, col=cfg.col).astype(np.float32)
        else:
            crop = utils.crop_zero_pad(cfg.img, x=cfg.col, y=cfg.row,
                                       w=int(self.crop_size * cfg.scale), h=int(self.crop_size * cfg.scale))
            # print(crop.shape, crop.dtype)
            crop = imresize(crop, [self.crop_size, self.crop_size], interp='bicubic')

        crop = crop / 255.0
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.25, r=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.5, r=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.7, r=cfg.brightness)

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        return crop * 255.0

    def gen_y(self, cfg: SampleCfg):
        img_scale = self.items_scale[cfg.train_idx]

        if self.use_categorical:
            return self.scale_to_one_shot(img_scale * cfg.scale)
        else:
            return img_scale * cfg.scale

    def generate(self, batch_size, is_training=True, max_extra_scale=1.4):
        step = -1
        pool = ThreadPool(processes=8)

        samples_to_process = []  # type: [SampleCfg]

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        images_to_keep = 8
        loaded_items = []
        test_img_id = 0
        while True:
            step += 1

            # keep 8 images loaded and replace one each 8 steps
            if len(loaded_items) < images_to_keep or step % 8 == 0:
                if len(loaded_items) == images_to_keep:
                    loaded_items.pop(0)

                if is_training:
                    item_to_load = random.choice(self.train_items)
                else:
                    item_to_load = self.test_items[test_img_id % len(self.test_items)]
                    test_img_id += 1

                loaded_items.append(dict(img=self.load_img(item_to_load[0]),
                                         train_idx=item_to_load[0],
                                         scale=item_to_load[1]))

            if is_training:
                item_to_process = random.choice(loaded_items)
                extra_scale = random.uniform(1.0 / max_extra_scale, max_extra_scale)

                img = item_to_process['img']
                train_idx = item_to_process['train_idx']
                cfg = SampleCfg(
                    img=img,
                    train_idx=train_idx,
                    row=random.randint(0, img.shape[0] - self.crop_size - 2),
                    col=random.randint(0, img.shape[1] - self.crop_size - 2),
                    contrast=rand_or_05(),
                    brightness=rand_or_05(),
                    saturation=rand_or_05(),
                    hflip=random.choice([True, False]),
                    vflip=random.choice([True, False]),
                    scale=extra_scale
                )
                samples_to_process.append(cfg)

            else:
                item_to_process = loaded_items[0]

                img = item_to_process['img']
                train_idx = item_to_process['train_idx']
                cfg = SampleCfg(
                    img=img,
                    train_idx=train_idx,
                    row=random.randint(0, img.shape[0] - self.crop_size - 2),
                    col=random.randint(0, img.shape[1] - self.crop_size - 2)
                )
                samples_to_process.append(cfg)

            if len(samples_to_process) == batch_size:
                x_batch = np.zeros((batch_size, self.crop_size, self.crop_size, 3))
                if self.use_categorical:
                    y_batch = np.zeros((batch_size, scale_categories), dtype=np.float32)
                else:
                    y_batch = np.zeros((batch_size, 1), dtype=np.float32)

                x_values = pool.map(self.gen_x, samples_to_process)
                for i, cfg in enumerate(samples_to_process):
                    x = x_values[i]
                    y = self.gen_y(cfg)
                    x_batch[i, :, :, :] = x
                    y_batch[i] = y

                samples_to_process = []
                yield self.preprocess_input(x_batch), y_batch
