from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import os.path as osp
from PIL import Image


class YoloDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, imsize=256.):
        self.transform = transform
        self.data_dir = data_dir
        self.imsize = imsize

        self.filenames = self.load_filenames(data_dir)

    def load_filenames(self, data_dir):
        filenames = [osp.join(data_dir, img) for img in os.listdir(data_dir) if os.path.splitext(img)[1] == '.png'
                     or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
        return filenames

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = self.load_img(filename)

        return img, filename

    def __len__(self):
        return len(self.filenames)
