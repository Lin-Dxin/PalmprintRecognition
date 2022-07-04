import torch
import model
import torch.utils.data as data
import torchvision.datasets as datasets
from Train import get_two_input_data
import torchvision.transforms as transforms
from Train import get_predict


def load_two_input_model():
    checkpoint = torch.load('./model1.pt')
    model_ = model.get_two_input_net()
    model_.load_state_dict(checkpoint)
    model_.eval()
    return model_


def loaddata(dir, img_size, batch_size_):
    dataset = datasets.ImageFolder(dir, transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_, shuffle=False)
    # for batch_data in loader:
    #     raw_input, raw_label = batch_data
    return loader


def get_labels(_labels, labels_final):
    pass


if __name__ == '__main__':
    # 读取模型
    test_model = load_two_input_model()
    batch__size = int(20)
    length = int(batch__size / 2)
    # 将图片转化为输入要求的格式：即两张照片的tensor
    # _input, _labels =
    correct_sum = 0
    for batch_data in loaddata(dir='Data/TestingSet/new_valdataset', img_size=[224, 224], batch_size_=batch__size):
        # 一个文件下有两个照片 刚好对应两个输入
        # 输出也是两个特征向量
        _input, _labels = batch_data
        input1 = torch.zeros([length, 3, 224, 224])
        input2 = torch.zeros([length, 3, 224, 224])
        labels_final = torch.zeros([length])
        i = 0
        j = 0
        # print(_input.shape)
        for it in range(0, batch__size, 2):
            input1[i] = _input[i]
            input2[i] = _input[i + 1]
            output1, output2 = test_model(input1, input2)
            print(output1)
            print(output2)
            print(_labels)
            for j in range(length):
                labels_final[j] = 1 if _labels[it] > 90 else -1
            print(labels_final)
            predict_result = get_predict(output1, output2)
            print(predict_result)
            # print(predict_result)
            i += 1
            break
        correct_sum += torch.sum(predict_result == labels_final.data)
        print(correct_sum)
        break
    print(correct_sum/180)
