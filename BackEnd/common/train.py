"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train.py
 @DateTime: 2023/4/23 16:02
 @SoftWare: PyCharm
"""

import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from LeNet5.net import LeNet5
from VGGNet.net import VGGNet


def load_net_opt_cri(model_name, device, lr, momentum):
    if model_name == 'LeNet5':
        net = LeNet5().to(device)
        if os.path.exists('../model/LeNet5/model.pth'):
            net.load_state_dict(torch.load('../model/LeNet5/model.pth'))
    elif model_name == 'VGGNet':
        net = VGGNet.to(device)
        net.load_state_dict(torch.load('../model/VGGNet/model.pth'))
    else:
        raise ValueError('model_name must be LeNet5 or VGGNet')

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # 加载优化器参数
    if model_name == 'LeNet5':
        if os.path.exists('../model/LeNet5/optimizer.pth'):
            optimizer.load_state_dict(torch.load('../model/LeNet5/optimizer.pth'))
    elif model_name == 'VGGNet':
        if os.path.exists('../model/VGGNet/optimizer.pth'):
            optimizer.load_state_dict(torch.load('../model/VGGNet/optimizer.pth'))

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    return net, optimizer, criterion


# 训练
def train(epochs, model_name, lr=0.001, momentum=0.9, batch_size=64):
    # 加载MNIST数据集
    if model_name == 'LeNet5':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif model_name == 'VGGNet':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 转换为3通道的灰度图像
            transforms.Resize(224),  # 调整大小为224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化
        ])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 使用GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 加载网络、优化器、损失函数
    net, optimizer, criterion = load_net_opt_cri(model_name, device, lr, momentum)
    print(net)

    # 训练网络
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))

            # 保存网络参数、优化器
            if model_name == 'LeNet5':
                torch.save(net.state_dict(), '../model/LeNet5/model.pth')
                torch.save(optimizer.state_dict(), '../model/LeNet5/optimizer.pth')
            elif model_name == 'VGGNet':
                torch.save(net.state_dict(), '../model/VGGNet/model.pth')
                torch.save(optimizer.state_dict(), '../model/VGGNet/optimizer.pth')
