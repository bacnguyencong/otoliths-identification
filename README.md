# Otolith identification

## Install

Install [conda](https://docs.conda.io/en/latest/miniconda.html) and run the following script for local installation

```bash
 git clone https://github.com/bacnguyencong/otoliths-identification
 cd otoliths-identification
 conda env create -f environment.yml
 conda activate otoliths-identification-env
```

## Data preprocessing

All references images should be located in `./data/Reference pictures/`. Run the following script to split data into training and valid sets:

```bash
python preprocess.py
```

There will be two folders `./data/train` and `./data/valid/` containing training and valid data, respectively.

Note:  Some images might not be well segmented due to the quality of images, please remove them to clean the data.

## Train the model

```bash
python main.py -img_size 224 -b 32 -j 4 --arch resnet18 -epochs 100 -lr_patience 5 -early_stop 10 -lr 0.001 --pretrain --train
```

## Predict the labels

```bash
python main.py --test
```

## TSNE

```bash
python make_tsne.py
```

## Highlight

```bash
python visualization.py
```


