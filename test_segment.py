
from util.useful_imports import *

# ## 2. Create test data

# Prepare a list of test images, containing all paths
img_list = []

for dire in os.listdir(SAMPLE_DIR):
    my_dir = os.path.join(SAMPLE_DIR, dire)
    for subdir in os.listdir(my_dir):        
        cur_dir = os.path.join(my_dir,subdir)
        if not os.path.isdir(cur_dir):
            continue
        #print(cur_dir)        
        jpg_list = glob.glob(cur_dir + '/*.jpg')
        tif_list = glob.glob(cur_dir + '/*.tif')
        
        img_list.extend(jpg_list)
        
        for tif_file in tif_list:
            name = os.path.basename(tif_file)
            name = name[0:-4]
            # check if format .jpg does not exist in the list
            if name + '.jpg' not in jpg_list:
                img_list.append(tif_file)


print("Segmenting the test data")
# Segment all images
from skimage.io import imread, imsave

if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
    
os.makedirs(TEST_DIR)

threshold = 0.25
remove_bg = False
conv_sigma = 2.0
opening_size = 30

for image_path in img_list:

    image = imread(image_path)    
    images_names = os.path.basename(image_path)
    images_names = images_names[0:-4] + '_{}.jpg'

    regions = ut.segment_image(image, remove_bg, threshold, conv_sigma,
                        opening_size)

    for i, reg in enumerate(regions):
        if len(images_names):
            min_row, min_col, max_row, max_col = reg.bbox
            segm_im = image[min_row:max_row][:,min_col:max_col]            
            imsave(os.path.join(TEST_DIR, images_names.format(i+1)), segm_im)
