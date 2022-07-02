import torch


IMAGE_SIZE = 244

USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = 'Trained ResNet.pth'