import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools

from skimage.io import imread, imsave
from skimage.color import rgb2grey, label2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction, binary_opening, watershed
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max




def sort_regions(regions):
    """
    Orders the segemented regions so that they match the numbering of a grid
    from top to bottom and from left to right.

    Input:
        - regions

    Output:
        - regions_sorted
    """
    regions_sorted = []
    regions.sort(key=lambda reg : reg.centroid[0], reverse=True)  # sort on rows

    regions_row = [regions.pop()]
    while regions:
        reg = regions.pop()  # new element for row
        min_row, min_col, max_row, max_col = regions_row[-1].bbox
        if reg.centroid[0] > min_row and reg.centroid[0] < max_row:  # part of row
            regions_row.append(reg)
        else:
            regions_sorted += sorted(regions_row, key=lambda reg : reg.centroid[1])
            regions_row = [reg]  # start new row
    regions_sorted += sorted(regions_row, key=lambda reg : reg.centroid[1])
    return regions_sorted


def segment_image(image, remove_bg=True, conv_sigma=2,
                    opening_size=30):
    """
    Segment larger regions from an image. Give it an image and it returns a
    list of region objects, approximately ordered accoring the numbering at the
    Excel file (i.e. from top to bottom and from left to right).

    Inputs:
        - image: greyscale image to segment
        - remove_bg: remove background? (bool, default: True)
        - conv_sigma: sigma of the Gaussian convolution (float, default: 2.0)
        - opening_size: size of the patch for binary opening (int, default: 30)

    Output:
        - regions: the segmented regions, sorted from top to bottom and from
                    left to right
    """
    image_greyscale = rgb2grey(image)
    # smoothing
    image_greyscale = gaussian_filter(image_greyscale, sigma=conv_sigma)
    # erosion
    seed = np.copy(image_greyscale)
    seed[1:-1, 1:-1] = image_greyscale.max()
    mask = image_greyscale
    dilated = reconstruction(seed, mask, method='erosion')
    # thresholding and binary opening
    image_thresholded = dilated > 0.5 * threshold_otsu(dilated)
    segmentation = binary_opening(image_thresholded,
                                    np.ones((opening_size, opening_size)))
    if 0:
        distance = ndi.distance_transform_edt(segmentation)
        local_maxi = peak_local_max(distance, indices=False,
                            footprint=np.ones((3, 3)),
                                    labels=segmentation)
        markers = ndi.label(local_maxi)[0]
        label_image = watershed(-distance, markers, mask=segmentation)
        segmentation = np.logical_or(label_image>0, segmentation)
    if remove_bg:
        image[segmentation==False] = 0
    label_image = label(segmentation)
    # find regions and sort them
    regions = sort_regions(regionprops(label_image))

    return regions

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

def loss_acc_plot(train, valid, label, output):
    """ Plot accuracies and losses over training
    Args:
        train: training input
        valid: validation input
        label:
    """
    fig = plt.figure()
    x = list(range(1, len(valid)+1))
    plt.plot(x, train, label='Train')
    plt.plot(x, valid, label='Valid')
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.xticks(x[0::(len(train)+9)//10])
    plt.legend()
    #plt.show()
    fig.savefig(output + label + '.png', bbox_inches='tight')

def plot_confusion_matrix(true_labels, pred_labels,
                          classes,
                          output,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    cm = confusion_matrix(true_labels, pred_labels, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(output+'confusion_matrix.png', bbox_inches='tight')

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
