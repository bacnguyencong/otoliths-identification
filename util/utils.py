import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

def imshow(image, title=None):
    """ show an image """    
    
    plt.imshow(image)
    plt.show()
    if title is not None:
        plt.title(title)


def image_to_tensor(image, mean=0, std=1.):
    """Transform image (input is numpy array, read in by cv2) 
    Args:
        image: a multiarray image from cv2 format
        mean:  the mean to normalize image
        std: the standard deviation to normalize image
    """
    
    image = image.astype(np.float32)
    image = (image-mean)/std
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
