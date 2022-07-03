import torch
from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet18_Weights


def fine_tune_model(use_gpu=False):
    model_feature = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # 记录全连接层的特征输入数
    num_features = model_feature.fc.in_features
    # 将最后一层的全连接层修改为我们定义的层
    model_feature.fc = nn.Linear(num_features, 2)
    if use_gpu:
        model_feature = model_feature.cuda()
    return model_feature


class two_input_net(nn.Module):
    def __init__(self, model):
        super(two_input_net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        # num_features = model.fc.in_features
        # self.linear = nn.Linear(num_features, 2)
        pass

    def forward(self, input1, input2):
        x1 = self.conv(input1)
        x2 = self.conv(input2)
        x1 = self.resnet_layer(x1)  # 512
        x2 = self.resnet_layer(x2)  # 512
        cos = nn.CosineSimilarity(dim=1, eps=1e-6) # 计算余弦相似度 = 1 - 距离
        # 取值范围 -1 ~ 1
        output = cos(x1, x2)
        return output


if __name__ == '__main__':
    model_feature = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # print(model_feature)
    model = two_input_net(model_feature)

    print(model)
