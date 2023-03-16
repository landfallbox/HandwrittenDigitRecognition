"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: net.py
 @DateTime: 2023/3/3 16:24
 @SoftWare: PyCharm
"""

import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个参数表示输入图像的通道数
        # 第二个参数表示卷积产生的通道数
        # kernel_size表示卷积核尺寸
        # 默认步长是1，默认填充是0
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 默认概率0.5
        self.conv2_drop = nn.Dropout2d()
        # 第一个参数表示输入的向量的维度
        # 第二个参数表示输出的向量的维度
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)
        # 把向量铺平。其中，-1表示不确定
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)
