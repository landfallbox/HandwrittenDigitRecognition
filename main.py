import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 整个数据集迭代次数
n_epochs = 3
# 一次训练或测试的规模
batch_size_train = 64
batch_size_test = 1000
# 学习率
learning_rate = 0.01
# SGD中的动量因子
momentum = 0.5
# 打印日志的间隔
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# 加载训练数据集，创建训练器
# 第一个参数：MNIST数据集的位置
# 第二个参数（可选）：True表示训练，False表示测试
# 第三个参数（可选）：True表示从网络上下载数据集，如果已存在，则不会重复下载
# 第四个参数（可选）：torchvision.transforms包含一些列图像预处理方法。其中，Compose串联多个图片变换的操作。
# ToTensor将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
# Normalize将图像标准化
dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                               torchvision.transforms.Normalize(
                                                                                   (0.1307,), (0.3081,))]))
# shuffle=True会在每个epoch重新打乱数据
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

# 加载测试数据集，创建测试器
dataset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                               torchvision.transforms.Normalize(
                                                                                   (0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, shuffle=True)


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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 把向量铺平。其中，-1表示不确定
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 初始化网络和优化器
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# 训练时的误差
train_losses = []
# 训练次数
train_counter = []
# 测试时的误差
test_losses = []
# 测试次数
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    # 设置网络为训练模式
    network.train()

    # 训练
    for batch_id, (data, target) in enumerate(train_loader):
        # 手动设置梯度为0
        optimizer.zero_grad()

        # 训练网络，得到预测值
        output = network(data)

        # 计算损失
        loss = F.nll_loss(output, target)

        # 将误差反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        if batch_id % log_interval == 0:
            # 打印训练日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_id / len(train_loader),
                                                                           loss.item()))
            # 保存训练的误差
            train_losses.append(loss.item())
            # 保存训练的次数
            train_counter.append((batch_id * 64) + ((epoch - 1) * len(train_loader.dataset)))
            # 保存网络模型参数
            torch.save(network.state_dict(), './model.pth')
            # 保存权重、偏置等
            torch.save(optimizer.state_dict(), './optimizer.pth')


train(1)





