"""
this file predict single image using the model we previous trained.
"""
from model import fine_tune_model
from global_config import *
import torch
import os
import sys
from data_loader.data_loader import DataLoader


def predict_single_image(inputs, classes_name):
    model = fine_tune_model()
    if not os.path.exists(MODEL_SAVE_FILE):
        print('can not find model save file.')
        exit()
    else:
        model.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=lambda storage, loc: storage))
        outputs = model(inputs)
        _, prediction_tensor = torch.max(outputs.data, 1)
        prediction = prediction_tensor.numpy()[0][0]
        # 增加与实际标签对比，记录准确率
        print('predict: ', prediction)
        print('this is {}'.format(classes_name[prediction]))


def predict():
    if len(sys.argv) > 1:
        print('predict image from : {}'.format(sys.argv[1]))  #文件路径
        data_loader = DataLoader(data_dir='datasets/hymenoptera_data', image_size=IMAGE_SIZE)
        if os.path.exists(sys.argv[1]):
            inputs = data_loader.make_predict_inputs(sys.argv[1])
            predict_single_image(inputs, data_loader.data_classes)
    else:
        print('must specific image file path.')

if __name__ == '__main__':
    predict()


