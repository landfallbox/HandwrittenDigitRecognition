"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: try.py
 @DateTime: 2023/4/27 11:05
 @SoftWare: PyCharm
"""
from torchvision import models

from VGGNet.net import VGGNet

# 自定义模型
model = VGGNet()
print("Custom model state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 预训练模型
pretrained_model = models.vgg16(pretrained=True)
print("Pretrained model state_dict:")
for param_tensor in pretrained_model.state_dict():
    print(param_tensor, "\t", pretrained_model.state_dict()[param_tensor].size())
