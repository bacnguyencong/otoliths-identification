import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

kernel_size = 2
depth_size1 = 3
depth_size2 = 3


class FineTuneModel(nn.Module):

    def __init__(self, original_model, arch, num_classes):

        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
        elif arch.startswith('densenet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Linear(original_model.classifier.in_features, num_classes)
            )
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            raise ("Finetuning not supported on this architecture yet")
        self.modelName = arch

    def forward(self, x):
        f = self.features(x)
        if self.modelName.startswith('densenet'):
            out = F.relu(f, inplace=True)
            out = F.avg_pool2d(out, kernel_size=7).view(f.size(0), -1)
            y = self.classifier(out)
        else:
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
        return y

class FineTuneModel_Hierarchical(nn.Module):

    def __init__(self, original_model, arch, args, gr_0_size=3, gr_1_size=3):

        super(FineTuneModel_Hierarchical, self).__init__()
        self.args = args
        
        if arch.startswith('alexnet'):
            self.features = original_model.features
            output_size = 256 * 6 * 6
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            output_size = original_model.fc.in_features
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)

        elif arch.startswith('densenet'):
            self.features = original_model.features
            output_size = original_model.classifier.in_features
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)

        elif arch.startswith('vgg16'):
            self.features = original_model.features
            output_size = 25088
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
        else:
            raise ("Finetuning not supported on this architecture yet")
        self.modelName = arch

    def forward(self, x):
        features = self.features(x)
        if self.modelName.startswith('densenet'):
            features = F.relu(f, inplace=True)
            features = F.avg_pool2d(out, kernel_size=7).view(f.size(0), -1)
        else:
            features = features.view(features.size(0), -1)
        return features

class CNNs(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), n_outputs=17):
        super(CNNs, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, depth_size1, kernel_size),
            nn.ReLU(),
            nn.Conv2d(depth_size1, depth_size2, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.flat_fts = self.get_flat_fts(input_shape, self.features)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, n_outputs),
            nn.Sigmoid()
        )

    def get_flat_fts(self, input_shape, fts):
        f = fts(Variable(torch.rand(1, *input_shape)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return out
