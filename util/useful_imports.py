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

from skimage.io import imread, imsave
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
SAMPLE_DIR = './data/Scheelhoek samples 2017'
REF_SEG_DIR = './data/Reference segmentation'

TEST_DIR = './data/test/'
TRAIN_DIR = './data/train/'
VALID_DIR = './data/valid/'
OUTPUT_FILE = './output/log.txt'
OUTPUT_WEIGHT_PATH = './output/'

GPU_AVAIL = torch.cuda.is_available()

