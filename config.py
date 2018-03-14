# -*- coding: utf-8 -*-
import torch

#ROOT_DIR = './data/images'
TEST_DIR = './data/test'
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'
OUTPUT_FILE = './output/log.txt'
OUTPUT_WEIGHT_PATH = './output/'
GPU_AVAIL = torch.cuda.is_available()
