# PalmRecognition
实现对人手掌纹的预测
# 要求

# 使用模型
ResNet-18

# 训练集与测试集
- 将session1作为训练集，session2作为测试集
- 可以将训练集做图像增强的操作，增加样本量（论文里采用了水平翻转）

# 训练
- 训练集在读取session1的所有图像上 + 一张session2读取到的第一张掌纹图 (mixed data mode)

# 模型细节
- ResNet对图片输入的大小要求是224 * 224
- 模型要输入RGB图像，训练数据是灰度图像，要将图像通道复制三次变成RGB图像
- 参数初始化：不随机初始化参数，而是将ImageNet预训练模型得到的参数作为初始参数
- 论文中得到的最优学习率是5 * 10^(-5)
- 采用深度学习网络学习得到掌纹的特征，输出掌纹特征，计算向量之间的相似度来判断是否为同一人

# 流程图
![1656581595484](https://user-images.githubusercontent.com/94331641/176644118-e4d9ad56-4129-4dd2-962b-ad1632ea1f1a.png)
from Biao Liu's paper
