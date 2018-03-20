import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def imshow(image, title=None):
    """ show an image """

    plt.imshow(image)
    plt.show()
    if title is not None:
        plt.title(title)

def make_square(img, fill_color=(0, 0, 0)):
    x, y = img.size
    size = max(x, y)
    new_img = Image.new(img.mode, (size, size), fill_color)
    new_img.paste(img, ((size - x) // 2, (size - y) // 2))
    return new_img

def crop_img(img, width, height, x=0, y=100):
    """ Crop a PIL image """
    area = (x, y, width, height)
    img = img.crop(area)
    return img

def resize_cv2(image, heigh=1280, width=1918):
    """ Resize of an cv2 image """
    return cv2.resize(image, (width, heigh), cv2.INTER_LINEAR)

def image_to_tensor(image):
    """Transform image (input is numpy array, read in by cv2)
    Args:
        image: a multiarray image from cv2 format
    """
    image = image.astype(np.float32)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)

    return tensor

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message: object, is_terminal: object = 1, is_file: object = 1) -> object:
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
