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
from torchvision import datasets

from common.load_net import load_net
from common.load_transform import load_transform
from common.args import loss_values

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 加载训练集
def load_train_loader(model_name, batch_size):
    transform = load_transform(model_name)

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


# 加载优化器
def load_optimizer(net, lr, momentum, model_info):
    optimizer = optim.SGD(net.parameters(), lr, momentum)

    if model_info is not None:
        optimizer.load_state_dict(model_info['optimizer_state_dict'])

    return optimizer


# 保存模型、优化器、损失、重启训练所需的参数
def save_model(net, optimizer, loss, epoch, save_path):
    print("Saving model...")

    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
    }, save_path)

    print("Model saved successfully to {}".format(save_path))


# 训练
def train(epochs, model_name, device, lr: float = 0.01, momentum: float = 0.9, batch_size: int = 64,
          resume: bool = False):
    print('Train device :' + device)

    # 加载训练集
    train_loader = load_train_loader(model_name, batch_size)

    # 模型保存路径
    save_path = '../model/' + model_name + '/model.pth'
    print('Model save path : ' + save_path)

    if os.path.exists(save_path):
        model_info = torch.load(save_path)
    else:
        model_info = None

    # 加载网络
    net = load_net(model_name, device, model_info)
    # print(net)
    net.train()

    # 加载优化器
    optimizer = load_optimizer(net, lr, momentum, model_info)

    # 加载保存的epoch
    if resume:
        start_epoch = model_info['epoch']
    else:
        start_epoch = 0

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 加载保存的损失
    if model_info is not None:
        saved_loss = model_info['loss']
    else:
        saved_loss = 10000.0

    # 训练网络
    for epoch in range(start_epoch, epochs):
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

                # 每100个样本记录一次loss值
                loss_values.append(loss.item())

                # 保存模型
                if loss.item() < saved_loss:
                    print('loss decrease from {:.4f} to {:.4f}'.format(saved_loss, loss.item()))
                    save_model(net, optimizer, loss.item(), epoch, save_path)

                    saved_loss = loss.item()

    # print(loss_values)

    # 绘制loss的变化图表
    plt.plot(range(len(loss_values)), loss_values)
    plt.title('Training Loss')
    plt.show()
