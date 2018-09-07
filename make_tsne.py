from sklearn.manifold import TSNE
import os
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

import util.utils as ut
import config as conf
from model.CNNs import FineTuneModel_Hierarchical
from torch.autograd import Variable

import matplotlib.pyplot as plt

model = models.__dict__['resnet18'](pretrained=True)
model = FineTuneModel_Hierarchical(model, 'resnet18', None)

# load the best model
checkpoint = torch.load(
    os.path.join(
        conf.OUTPUT_WEIGHT_PATH, 'best_{}.pth.tar'.format(
            model.modelName
        )
    ),
    map_location=lambda storage, loc: storage
)
model.load_state_dict(checkpoint['state_dict'])
model.args = checkpoint['args']
model.eval()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
valid_trans = transforms.Compose([
    transforms.Lambda(lambda x: ut.make_square(x)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# data loader for training
dset_train = ImageFolder(root=conf.TRAIN_DIR, transform=valid_trans)
train_loader = DataLoader(
    dset_train,
    batch_size=4,
    shuffle=True,
    num_workers=1,
    pin_memory=0
)

X_train, y_train = [], []  # type: ignore

for batch_idx, (inputs, y) in enumerate(train_loader):
    input_var = Variable(inputs)
    batch_size = inputs.size(0)
    # forward net
    outputs = model(input_var)
    X_train.append(outputs.data.numpy())
    y_train.extend(y.data.numpy())

imgs = np.vstack(X_train)
tsne = TSNE(n_components=2, random_state=1234)
X = tsne.fit_transform(imgs)

cl_coding = ['green', 'cyan', 'blue', 'red', 'pink', 'orange']
color_map = {
    model.args['idx_to_lab'][i]: cl_coding[i]
    for i in range(len(cl_coding))
}

df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['Otolith type'] = np.array([model.args['idx_to_lab'][i] for i in y_train])

axes = sns.scatterplot(
    x="Feature 1",
    y="Feature 2",
    hue="Otolith type",
    data=df,
    palette=color_map
)

plt.savefig(
    os.path.join(
        conf.OUTPUT_WEIGHT_PATH,
        'tSNE_trained.png'
    ),
    format='png',
    bbox_inches='tight',
    dpi=500
)
