
# coding: utf-8

# # Otoliths predictions

# In[1]:

get_ipython().magic('matplotlib inline')
from util.useful_imports import *


# # Data processing

# ##  1. Partitioning data into training and test
# ### 1.1. Segment the training


if os.path.exists(REF_SEG_DIR):
    shutil.rmtree(REF_SEG_DIR)
    
os.makedirs(REF_SEG_DIR)

threshold = 0.25
remove_bg = True
conv_sigma = 2.0
opening_size = 30

for label in os.listdir(ROOT_DIR):
    
    cur_dir = os.path.join(ROOT_DIR, label)
    tar_dir = os.path.join(REF_SEG_DIR, label)
    
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    
    img_list = glob.glob(cur_dir + '/*.jpg')
    
    print(label)
    
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
                
                if segm_im.size / 8192 < 20.0: #this are noise
                    continue                    
                if segm_im.shape[0] == 1536 and segm_im.shape[1] == 2048: # cannot segment
                    continue 
                imsave(os.path.join(tar_dir, images_names.format(i+1)), segm_im)


# Partition the data to train and valid
# create train and valid directories
if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)
os.makedirs(TRAIN_DIR)

if os.path.exists(VALID_DIR):
    shutil.rmtree(VALID_DIR)    
os.makedirs(VALID_DIR)

train_per = 0.9
# making a partition of training and valid sets
for dire in os.listdir(REF_SEG_DIR):
    
    #create path for train
    p1 = os.path.join(TRAIN_DIR, dire)
    if not os.path.exists(p1):
        os.makedirs(p1)
        
    #create path for valid
    p2 = os.path.join(VALID_DIR, dire)
    if not os.path.exists(p2):
        os.makedirs(p2)
    
    img_list = glob.glob(os.path.join(REF_SEG_DIR, dire) + '/*.jpg')
    n = len(img_list)
    rp = np.random.permutation(n)

    # number of training images
    train = math.floor(train_per * n)
    
    for i in range(n):
        j = rp[i]
        
        filepath = img_list[j]
        filename = os.path.basename(filepath)
        
        if i < train or n <= 5:
            copyfile(filepath, p1 + '/' + filename)
        
        if i >= train or n <= 5:
            copyfile(filepath, p2 + '/' + filename)

#delete the folder
shutil.rmtree(REF_SEG_DIR)

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


# Segment all images
from skimage.io import imread, imsave

if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
    
os.makedirs(TEST_DIR)

threshold = 0.25
remove_bg = True
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
