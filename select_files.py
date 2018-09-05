from shutil import copyfile

import pandas as pd

from util.useful_imports import *

df = pd.read_excel('data/Data_Delta_schalen samples_faeces adult Sandwich Tern_2017_cleaned.xlsx')

img_names = df[['Picture_ID', 'Nr_on_picture']].apply(lambda x: ''.join(x[0] + '_' + str(x[1]) + '.jpg'), axis=1)
img_names = img_names.apply(lambda x: x[0:5] + 'AD' + '_' + x[8:])


if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

os.makedirs(TEST_DIR)

num = 0
for it in img_names.values:
    it = it[2:]
    file = 'data/test_all/' + it
    if os.path.isfile(file):
        num += 1
        copyfile(file, TEST_DIR + 'F_' + it[0:3] + 'ad' + it[5:])

print('Extracting %d/%d files' % (num, len(img_names)))
