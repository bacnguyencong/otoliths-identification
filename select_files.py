from util.useful_imports import *
import pandas as pd
from shutil import copyfile


df = pd.read_excel('data/Data_Delta_schalen samples_faeces adult Sandwich Tern_2017_cleaned.xlsx')

img_names = df[['Picture_ID', 'Nr_on_picture']].apply(lambda x: ''.join(x[0] + '_' + str(x[1]) + '.jpg'), axis=1)
img_names = img_names.apply(lambda x: x[0:5] + 'AD' + '_' + x[8:])

from shutil import copyfile

if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

os.makedirs(TEST_DIR)

for it in img_names.values:
    file = 'test_all/' + it
    if os.path.exists(file):
        copyfile(file, TEST_DIR + it)

