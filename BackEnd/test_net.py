"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: test_net.py
 @DateTime: 2023/3/3 16:29
 @SoftWare: PyCharm
"""

import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn.functional as f

from net import Net

# 一次测试的规模
batch_size_test = 1000

# 如果GPU可用则使用GPU训练和测试。否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
network = Net().to(device)

# 加载测试数据集，创建测试器
dataset_test = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                    torchvision.transforms.Normalize(
                                                                                        (0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)

# 加载网络模型和参数
network.load_state_dict(torch.load('./model.pth'))

# 测试模式
network.eval()

# 初始化
test_loss = 0
correct = 0

# 测试时的误差
test_losses = []

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
print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                          correct, len(dataset_test),
                                                                          100. * correct / len(dataset_test)))
