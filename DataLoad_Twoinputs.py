import numpy as np
import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.nn.parallel
# import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.transforms
import torchvision.transforms as transforms
from PIL import Image
# import glob
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable


class DataLoader(object):
    def __init__(self, data_dir, image_size, batch_size=50):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.CenterCrop(self.image_size),
                # transforms.CenterCrop([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
        }
        self._init_data_sets()

    def _init_data_sets(self):
        self.data_sets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}

        self.data_loaders = {x: torch.utils.data.DataLoader(self.data_sets[x], batch_size=self.batch_size,
                                                            shuffle=False, num_workers=4)
                             for x in ['train', 'val']}
        self.data_sizes = {x: len(self.data_sets[x]) for x in ['train', 'val']}
        self.data_classes = self.data_sets['train'].classes

    def load_data(self, data_set='train'):
        return self.data_loaders[data_set]

    def show_image(self, tensor, title=None):
        inp = tensor.numpy().transpose((1, 2, 0))
        # put it back as it solved before in transforms
        inp = self.normalize_std * inp + self.normalize_mean
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.show()

    def make_predict_inputs(self, image_file):
        """
        this will make a image to PyTorch inputs, as the same with training images.
        this will return a tensor, default not using CUDA.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_tensor = self.data_transforms['val'](image).float()
        image_tensor.unsqueeze_(0)
        return Variable(image_tensor)


