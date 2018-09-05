import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch
#import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset

import util.utils as ut


class DataLoader(Dataset):
    """Fish dataset"""

    def __init__(self, csv_file, root_dir, transform=None, classes=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        if csv_file:
            meta_info = pd.read_csv(csv_file)
            self.images = meta_info['image']
            self.classes = meta_info['label'].unique() if classes is None else classes
            self.num_classes = len(self.classes)
            self.encoder = LabelEncoder().fit(self.classes)
            self.labels = self.encoder.transform(meta_info['label'])
        else:
            self.images = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
            self.labels = [-1] * len(self.images)

        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.images[idx])
        #image = cv2.imread(img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return {'image': image, 'label': label, 'name': self.images[idx]}

    def __len__(self):
        return len(self.labels)


class DataLoaderFromPILL(Dataset):
    """Fish dataset"""

    def __init__(self, PIL_img_list, transform=None, classes=None):
        """
        Args:
            PIL_img_list (PIL.Image): List of images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = PIL_img_list
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': -1}

    def __len__(self):
        return len(self.images)
