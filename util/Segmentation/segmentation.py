"""
Created on Monday 19 March 2018
<<<<<<< HEAD
Last update: Tuesday 20 March 2018
=======
Last update: -
>>>>>>> 7420f84b5cd0a983f1a4af1d6937c4b8dbb42a06

@author: Michiel Stock
michielfmstock@gmail.com

Module to segment the otoliths (or other things) from an image.
"""


from skimage.io import imread, imsave
from skimage.io import imread

from skimage.color import rgb2grey, label2rgb
from skimage.morphology import reconstruction, binary_opening
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
import numpy as np

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


def segment_image(image, remove_bg=True, threshold=0.25, conv_sigma=2,
                    opening_size=30):
    """
    Segment larger regions from an image. Give it an image and it returns a
    list of region objects, approximately ordered accoring the numbering at the
    Excel file (i.e. from top to bottom and from left to right).

    Inputs:
        - image: greyscale image to segment
        - remove_bg: remove background? (bool, default: True)
        - threshold: threshold (float, default: 0.25)
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
    segmentation = binary_opening(dilated > threshold,
                                    np.ones((opening_size, opening_size)))
    if remove_bg:
        image[segmentation==False] = 0
    label_image = label(segmentation)
    # find regions and sort them
    regions = sort_regions(regionprops(label_image))
    return regions



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Segment the otoliths from an image.')
    parser.add_argument('image', type=str, help='path to the image')
    parser.add_argument('--threshold', help='threshold', type=float, default=0.25)
    parser.add_argument('--remove_bg', action='store_true')
    parser.add_argument('--conv_sigma', type=float, default=2,
                        help='sigma of the Gaussian convolution')
    parser.add_argument('--opening_size', type=int, default=30,
                        help='size of the patch for binary opening')
    parser.add_argument('--images_names', type=str, default='',
                        help="""
                        name for segmented images that are cut, should be a formatter
                        of the form

                        log/image{}.jpg
                        """)

    args = parser.parse_args()

    image_path = args.image
    image = imread(image_path)
    threshold = args.threshold
    remove_bg = args.remove_bg
    conv_sigma = args.conv_sigma
    opening_size = args.opening_size
    images_names = args.images_names

    regions = segment_image(image, remove_bg, threshold, conv_sigma,
                        opening_size)

    for i, reg in enumerate(regions):
        print(reg.bbox)
        if len(images_names):
            min_row, min_col, max_row, max_col = reg.bbox
            segm_im = image[min_row:max_row][:,min_col:max_col]
            imsave(images_names.format(i+1), segm_im)
#TODO: implement argtools
