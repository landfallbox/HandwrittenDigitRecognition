"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: test.py
 @DateTime: 2023/4/23 16:09
 @SoftWare: PyCharm
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from LeNet5.net import LeNet5
from VGGNet.net import VGGNet


# 测试
def test(model_name):
    # 设置超参数
    batch_size = 64

    # 使用GPU测试
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'LeNet5':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # 初始化LeNet-5网络，加载网络参数
        net = LeNet5().to(device)
        net.load_state_dict(torch.load('../model/LeNet5/model.pth'))
    elif model_name == 'VGGNet':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 转换为3通道的灰度图像
            transforms.Resize(224),  # 调整大小为224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化
        ])
        # 初始化VGGNet网络，加载网络参数
        net = VGGNet.to(device)
        net.load_state_dict(torch.load('../model/VGGNet/model.pth'))

    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
