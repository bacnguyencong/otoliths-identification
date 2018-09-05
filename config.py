# -*- coding: utf-8 -*-
import torch
import os

#ROOT_DIR = './data/images'
TEST_DIR = './data/test'
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'
OUTPUT_FILE = './output/log.txt'
TEST_FILE = './data/Data_Delta_schalen samples_faeces adult Sandwich Tern_2017_cleaned.xlsx'

OUTPUT_WEIGHT_PATH = './output/'

if not os.path.exists(OUTPUT_WEIGHT_PATH):
        os.makedirs(OUTPUT_WEIGHT_PATH)
        
GPU_AVAIL = torch.cuda.is_available()

INPUT_TEST_DIR = './data/Scheelhoek samples 2017/'
OUTPUT_TEST_DIR = './output/labeled samples 2017/'
