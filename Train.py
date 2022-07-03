import copy

import torch
import torchvision
import torch.nn as nn
from torchvision.models import ResNet18_Weights

import model
import DataLoad
import time
import torch.optim as optim
from torch.autograd import Variable
from random import sample
from torchvision import models


def train_model(data_loader, model, criterion, optimizer, lr_scheduler, num_epochs=25, use_gpu=False):
    """
    the pipeline of train PyTorch model
    :param use_gpu:
    :param data_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param lr_scheduler:
    :param num_epochs:
    :return:
    """
    since_time = time.time()

    best_model = model
    best_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for batch_data in data_loader.load_data(data_set=phase):
                rawinputs, rawlabels = batch_data
                input1, input2, labels = get_two_input_data(rawinputs, rawlabels, data_loader.batch_size)
                # data_loader.show_image(input1[0])
                if use_gpu:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    input1 = input1.to(device)
                    input2 = input2.to(device)
                    labels = labels.to(device)
                else:
                    input1, input2, labels = Variable(input1), Variable(input2), Variable(labels)
                # print(inputs.shape, labels.shape)
                # print(labels)
                # inputs_num += len(inputs)
                optimizer.zero_grad()

                outputs = model(input1, input2)
                # print(outputs.data)  # 看一下余弦相似度的范围
                # _, predict = torch.max(outputs.data, 1)
                # 设置一个判断，predict=1 if output.data>0.5 else 0
                predict = get_predict(outputs)
                predict = predict.cuda()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # collect data info
                running_loss += loss.data
                running_corrects += torch.sum(predict == labels.data)

            epoch_loss = running_loss / (data_loader.data_sizes[phase] * 2)
            epoch_acc = running_corrects / (data_loader.data_sizes[phase] * 2)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def get_predict(outputs):
    # print(outputs.shape)
    predict = torch.zeros(len(outputs), dtype=torch.int64)
    i = 0
    for result in outputs:
        if result > 0.5:
            predict[i] = 1
        else:
            predict[i] = 0
        i += 1
    return predict


def get_two_input_data(rawinputs, rawlabels, batch_size):
    length = int(batch_size / 2)
    it_times = int(batch_size / 2)  # 100/2=50
    input1 = torch.zeros([length, 3, 244, 244], dtype=torch.float32)  # [50,3,244,244]
    input2 = torch.zeros([length, 3, 244, 244], dtype=torch.float32)
    labels = torch.zeros(length, dtype=torch.float32)
    cnt = 0
    j = 0
    for i in range(it_times):  # 50
        if i < it_times / 2:  # <25
            input1[j] = rawinputs[cnt] # 0 1
            label1 = rawlabels[cnt]
            cnt += 1
            input2[j] = rawinputs[cnt]
            label2 = rawlabels[cnt]
            cnt += 1
            labels[j] = 1 if label1 == label2 else 0
            j += 1
        else:  # >25
            rand_a, rand_b = sample(range(it_times, batch_size, 1), 2)
            input1[j] = rawinputs[rand_a]
            label1 = rawlabels[rand_a]
            input2[j] = rawinputs[rand_b]
            label2 = rawlabels[rand_b]
            labels[j] = 1 if label1 == label2 else 0
            j += 1
    # print(input1.shape,"\n",input2.shape,"\n",labels.shape)
    return input1, input2, labels


def get_concated_data(rawinputs, rawlabels, batch_size):
    # 二和一、每次往后迭代两次
    # 前半部分不打乱、后半部分打乱
    # cnt 记录拼接次数、一旦超过一半后
    # 随机挑选后面的数字 6000 3000 3000 1500 1500
    # 3000 if cnt > 1500: random_pic (1500,3000)
    length = int(batch_size / 2)
    inputs = torch.zeros([length, 3, 244, 244], dtype=torch.float32)
    labels = torch.zeros(length, dtype=torch.int64)
    # print(rawlabels.dtype)
    # print(rawinputs.dtype)
    it_times = int(batch_size / 2)
    cnt = 0
    j = 0
    for i in range(it_times):  # 0  2  4
        if i < it_times / 2:
            inputs[j] = torch.cat([rawinputs[cnt], rawinputs[cnt + 1]], 1)
            labels[j] = 1 if rawlabels[cnt] == rawlabels[cnt + 1] else 0
            # print(rawlabels[cnt], rawlabels[cnt + 1])
            # print("=======label:",j,"++++",labels[j],"=======")
            j = j + 1

            # 顺序抽取
        else:
            rand_a, rand_b = sample(range(it_times, batch_size, 1), 2)  # 从后半部分随机抽取两个数字
            inputs[j] = torch.cat([rawinputs[rand_a], rawinputs[rand_b]], 1)
            labels[j] = 1 if rawlabels[rand_a] == rawlabels[rand_b] else 0
            # print(rawlabels[rand_a], rawlabels[rand_b])
            # print("=======label:", j, "++++", labels[j], "=======")
            j = j + 1
            # 随机抽取
        cnt = cnt + 2
    return inputs, labels


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def save_torch_model(model, name):
    torch.save(model.state_dict(), name)


if __name__ == '__main__':
    train_dir = "data/TrainingSet/NIR/"
    _batchsize = 100
    data_loader = DataLoad.DataLoader(data_dir='Data/TrainingSet/ROI_image/', image_size=244,
                                      batch_size=_batchsize)
    # print(data_loader.data_sizes['train'])
    # inputs, classes = next(iter(data_loader.load_data()))
    # out = torchvision.utils.make_grid(inputs)
    # data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])

    # model = model.fine_tune_model(use_gpu=True)

    model = model.get_two_input_net()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()  # 尝试换不同的损失函数
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    try:
        model = train_model(data_loader, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, use_gpu=True)
        save_torch_model(model, "model1")
    except KeyboardInterrupt:
        print('manually interrupt, try saving model for now...')
        save_torch_model(model, "model1")
        print('model saved.')
