import glob
import math
import os
import random
import shutil
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
from skimage.io import imread, imsave
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from tqdm import tqdm

import util.data_utils as pu
import util.utils as ut
from model import CNNs

ROOT_DIR = './data/Reference pictures/'
SAMPLE_DIR = './data/Scheelhoek samples 2017'
REF_SEG_DIR = './data/Reference segmentation'

TEST_DIR = './data/test/'
TRAIN_DIR = './data/train/'
VALID_DIR = './data/valid/'
OUTPUT_FILE = './output/log.txt'
OUTPUT_WEIGHT_PATH = './output/'

GPU_AVAIL = torch.cuda.is_available()
