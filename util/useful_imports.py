import matplotlib.pyplot as plt

import os
import glob
from shutil import copyfile
import math
import shutil

import torch
import numpy as np
import random
import util.data_utils as pu
import util.utils as ut
import pandas as pd
import PIL

from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import util.utils as ut

from tqdm import tqdm
import torch.nn as nn
from model import CNNs
from torch.autograd import Variable

ROOT_DIR = './data/Reference pictures/'
TEST_DIR = './data/test/'
TRAIN_DIR = './data/train/'
VALID_DIR = './data/valid/'
OUTPUT_FILE = './output/log.txt'
OUTPUT_WEIGHT_PATH = './output/'
GPU_AVAIL = torch.cuda.is_available()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
valid_trans = transforms.Compose([
        transforms.Lambda(lambda x: ut.crop_img(x, 2000, 1300)),
        transforms.Lambda(lambda x: ut.make_square(x)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
])
dset_valid = ImageFolder(root='data/valid/', transform=valid_trans)
valid = transforms.Compose([
        transforms.Lambda(lambda x: ut.crop_img(x, 2000, 1300)),
        transforms.Lambda(lambda x: ut.make_square(x)),
        transforms.Resize((224, 224))
])
dsetreal = ImageFolder(root='data/valid/', transform=valid)


data = ImageFolder(root=TRAIN_DIR,
                   transform=
                       transforms.Compose([
                           transforms.Lambda(lambda x: ut.crop_img(x, 2000, 1300)),
                           transforms.Lambda(lambda x: ut.make_square(x)),
                           transforms.Resize((224, 224)),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(45,PIL.Image.BILINEAR)])
                          )
