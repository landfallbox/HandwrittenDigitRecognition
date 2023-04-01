"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: net.py
 @DateTime: 2023/3/3 16:24
 @SoftWare: PyCharm
"""

import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.functional as f
import torch.optim as optim


# 网络结构
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


# 训练相关变量
# 整个数据集迭代次数
epochs = 3
# 一次训练的规模
batch_size_train = 64
# 学习率
learning_rate = 0.01
# SGD中的动量因子
momentum = 0.5
# 训练时的误差
train_losses = []
# 训练次数
train_counter = []

# 打印日志的间隔
log_interval = 10

# 随机种子
random_seed = 1
torch.manual_seed(random_seed)

# 如果GPU可用则使用GPU训练和测试。否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
network = Net().to(device)

# 加载训练数据集，创建训练器
# 第一个参数：MNIST数据集的位置
# 第二个参数（可选）：True表示训练，False表示测试
# 第三个参数（可选）：True表示从网络上下载数据集，如果已存在，则不会重复下载
# 第四个参数（可选）：torchvision.transforms包含一些列图像预处理方法。其中，Compose串联多个图片变换的操作。
# ToTensor将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
# Normalize将图像标准化
# RandomRotation随机旋转
dataset_train = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                torchvision.transforms.RandomRotation(15)
                                                ])
                                           )
# shuffle=True会在每个epoch重新打乱数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)

# 初始化优化器
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


# 训练网络
def train(epoch):
    # 加载网络模型和参数
    if os.path.exists('./model.pth'):
        network.load_state_dict(torch.load('./model.pth'))

    # 加载优化器
    if os.path.exists('./optimizer.pth'):
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
        # nll_loss与CrossEntropyLoss的区别在于，CrossEntropyLoss会自动添加softmax层，而nll_loss不会
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


# 测试相关变量
# 一次测试的规模
batch_size_test = 1000
# 测试次数
test_counter = []
# 测试时的误差
test_losses = []

# 加载测试数据集，创建测试器
dataset_test = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)


# 测试网络
def test():
    # 加载网络模型和参数
    network.load_state_dict(torch.load('./model.pth'))

    # 测试模式
    network.eval()

    # 初始化
    test_loss = 0
    correct = 0

    # 关闭反向传播时的自动求导，节约内存
    with torch.no_grad():
        for data, target in test_loader:
            # 数据和标签加载到GPU上
            data = data.to(device)
            target = target.to(device)

            # 用训练好的网络得到预测值
            output = network(data)
            # 一个batch中的误差相加
            test_loss += f.nll_loss(output, target, reduction='mean').item()
            # 找到最大值的下标
            pred = output.data.max(1, keepdim=True)[1]
            # 每判断正确一个结果，correct加一
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(dataset_test)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                              correct, len(dataset_test),
                                                                              100. * correct / len(dataset_test)))
