
from torchvision import models
from torch import nn
from torchvision.models import ResNet18_Weights


def fine_tune_model():
    model_feature = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # 记录全连接层的特征输入数
    num_features = model_feature.fc.in_features
    # 将最后一层的全连接层修改为我们定义的层
    model_feature.fc = nn.Linear(num_features, 2)

    return model_feature
