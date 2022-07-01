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
                                 transforms.Resize([122, 244]),
                                 transforms.RandomResizedCrop([122, 244]),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize([122, 244]),
                                 transforms.CenterCrop(244),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


if __name__ == '__main__':
    train_dir = "data/TrainingSet/NIR"
    test_ld = loader(train_dir)
    for data in test_ld:
        rawInput = data[0]  # 数据文件
        rawLabel = data[1]  # 标签
        label = torch.zeros(16)
        for i in range(16):
            x = torch.cat((rawInput[i], rawInput[i + 1]), 1)
            label[i] = 1 if rawLabel[i] == rawLabel[i + 1] else 0
            i += 2
        print(x.shape, label.shape)
        print(label, rawLabel)
        break




    # list_plastic = os.listdir(train_dir)
    # number_files_plastic = len(list_plastic)
    # print(number_files_plastic)
