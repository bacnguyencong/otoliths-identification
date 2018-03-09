# -*- coding: utf-8 -*-
import torch

ROOT_DIR = './data/images'
TEST_DIR = './data/images'
TRAIN_CSV_FILE = './data/data.csv'
VALID_CSV_FILE = './data/data.csv'
OUTPUT_FILE = './output/log.txt'
OUTPUT_WEIGHT_PATH = './output/'
GPU_AVAIL = torch.cuda.is_available()
