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


# 测试
def test(model_name):
    # 设置超参数
    batch_size = 64

    # 加载测试数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 使用GPU测试
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'LeNet5':
        # 初始化LeNet-5网络，加载网络参数
        net = LeNet5().to(device)
        net.load_state_dict(torch.load('../model/LeNet5/model.pth'))
    elif model_name == 'VGGNet':
        # 初始化VGGNet网络，加载网络参数
        net = models.vgg16(pretrained=False).to(device)
        net.load_state_dict(torch.load('../model/VGGNet/model.pth'))

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
