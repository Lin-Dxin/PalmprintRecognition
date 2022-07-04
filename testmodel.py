import torch
import model
import torch.utils.data as data
import torchvision.datasets as datasets
from Train import get_two_input_data
import torchvision.transforms as transforms
from Train import get_predict


def load_two_input_model():
    checkpoint = torch.load('./model1')
    model_ = model.get_two_input_net()
    model_.load_state_dict(checkpoint)
    model_.eval()
    return model_


def loaddata(dir, img_size, batch_size_):
    dataset = datasets.ImageFolder(dir, transforms.Compose([
        transforms.Resize(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_, shuffle=False)
    for batch_data in loader:
        raw_input, raw_label = batch_data
    return raw_input, raw_label


if __name__ == '__main__':
    # 读取模型
    test_model = load_two_input_model()
    batch__size = int(20)
    # 将图片转化为输入要求的格式：即两张照片的tensor
    _input, _labels = loaddata(dir='Data/TestingSet/new_valdataset', img_size=[224, 224], batch_size_=batch__size)
    # 一个文件下有两个照片 刚好对应两个输入
    # 输出也是两个特征向量
    sum = 0
    input1 = torch.zeros([10, 3, 224, 224])
    input2 = torch.zeros([10, 3, 224, 224])
    labels_final = torch.zeros([10])
    i = 0
    for pic in _input:
        input1[i] = pic[0]
        input2[i] = pic[1]
        labels_final[i] = 1 if _labels[i] < 90 else -1

        output1, output2 = test_model(input1, input2)
        # predict_result 是模型预测的结果，一次有 batchsize / 2 个
        predict_result = get_predict(output1, output2)
        # 检测真实结果predict_result与预测结果_labels  求和正确数
        for j in range(10):
            if predict_result[j] == labels_final[j]:
                sum += 1
        print("正确预测数为：", sum)
        i += 1
