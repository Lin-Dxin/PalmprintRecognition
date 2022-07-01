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
# from PIL import Image
# import glob
import os


def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.RandomResizedCrop(256),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

if __name__ == '__main__':
    train_dir = "data/TrainingSet/NIR"
    test_ld = test_loader(train_dir)
    for data in test_ld:
        x = data[0] # 数据文件
        y = data[1] # 标签
        print("数据内容为：\n", x.shape)
        print("数据标签为：\n", y.shape)
        break

    # list_plastic = os.listdir(train_dir)
    # number_files_plastic = len(list_plastic)
    # print(number_files_plastic)