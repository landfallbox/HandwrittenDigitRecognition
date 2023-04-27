"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: net.py
 @DateTime: 2023/4/26 21:42
 @SoftWare: PyCharm
"""
from torch import nn
from torchvision import models


# 使用预训练的VGGNet16搭建自己的VGGNet，在预训练网络后面添加一个全连接层
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.pre_trained = models.vgg16(pretrained=True)
        for param in self.pre_trained.parameters():
            param.requires_grad = False
        self.classififer = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pre_trained(x)
        x = self.classififer(x)
        return x
