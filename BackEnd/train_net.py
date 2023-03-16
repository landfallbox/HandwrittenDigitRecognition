"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train_net.py
 @DateTime: 2023/3/1 17:58
 @SoftWare: PyCharm
"""

import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn.functional as f
import torch.optim as optim

from net import Net

# 整个数据集迭代次数
n_epochs = 3
# 一次训练的规模
batch_size_train = 64

# 学习率
learning_rate = 0.01
# SGD中的动量因子
momentum = 0.5
# 打印日志的间隔
log_interval = 10

# random_seed = 1
# torch.manual_seed(random_seed)

# 加载训练数据集，创建训练器
# 第一个参数：MNIST数据集的位置
# 第二个参数（可选）：True表示训练，False表示测试
# 第三个参数（可选）：True表示从网络上下载数据集，如果已存在，则不会重复下载
# 第四个参数（可选）：torchvision.transforms包含一些列图像预处理方法。其中，Compose串联多个图片变换的操作。
# ToTensor将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
# Normalize将图像标准化
dataset_train = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                torchvision.transforms.RandomRotation(15)
                                                ])
                                           )
# shuffle=True会在每个epoch重新打乱数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)

# 如果GPU可用则使用GPU训练和测试。否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
network = Net().to(device)

# 初始化优化器
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# 训练时的误差
train_losses = []
# 训练次数
train_counter = []


def train(epoch):
    # 加载网络模型和参数
    network.load_state_dict(torch.load('./model.pth'))
    # 加载优化器
    optimizer.load_state_dict(torch.load('./optimizer.pth'))

    # 设置网络为训练模式
    network.train()

    # 训练
    for batch_id, (data, target) in enumerate(train_loader):
        # 数据和标签加载到GPU上
        data = data.to(device)
        target = target.to(device)

        # 手动设置梯度为0
        optimizer.zero_grad()

        # 训练网络，得到预测值
        output = network(data)

        # 计算损失
        loss = f.nll_loss(output, target).to(device)

        # 将误差反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        if batch_id % log_interval == 0:
            # 打印训练日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id * len(data),
                                                                           len(dataset_train),
                                                                           100. * batch_id / len(train_loader),
                                                                           loss.item()))
            # 保存训练的误差
            train_losses.append(loss.item())
            # 保存训练的次数
            train_counter.append((batch_id * 64) + ((epoch - 1) * len(dataset_train)))
            # 保存网络模型和参数
            torch.save(network.state_dict(), './model.pth')
            # 保存优化器
            torch.save(optimizer.state_dict(), './optimizer.pth')


# 多次迭代训练神经网络
n = 1
while n <= n_epochs:
    train(n)
    n += 1
