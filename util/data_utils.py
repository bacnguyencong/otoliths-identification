import os
import cv2
import pandas as pd
import numpy as np
import util.utils as ut
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder

class DataLoader(Dataset):
    """Fish dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None, classes=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        meta_info = pd.read_csv(csv_file)
        self.images = meta_info['image']
        self.classes = meta_info['label'].unique() if classes is None else classes
        self.num_classes = len(self.classes)
        self.encoder = LabelEncoder().fit(self.classes)
        self.labels = self.encoder.transform(meta_info['label'])
        self.root_dir = root_dir
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.images[idx])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        image = ut.image_to_tensor(image/255)

        return {'image': image, 'label': label, 'name': self.images[idx]}

    def __len__(self):
        return len(self.labels)
