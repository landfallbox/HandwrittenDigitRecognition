"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: test.py
 @DateTime: 2023/4/23 16:09
 @SoftWare: PyCharm
"""
import os

import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets

from common.load_transform import load_transform
from common.train import load_net


# 加载测试集
def load_test_loader(model_name, batch_size):
    transform = load_transform(model_name)

    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


# 测试
def test(model_name, device, batch_size: int = 100):
    # 加载测试集
    test_loader = load_test_loader(model_name, batch_size)

    # 加载模型
    net = load_net(model_name, device, torch.load('../model/' + model_name + '/model.pth'))
    net.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        # processed_samples = 0

        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # if total % 1000 == 0:
            #     # print('Accuracy of the network on the {} test images: {} %'.
            #     #       format((i + 10) * 100, 100 * correct / total))
            #
            #     accuracy = 100 * correct / total
            #     print('Accuracy after {} samples: {} %'.format(total, accuracy))

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        print("Confusion Matrix:")
        print(cm)

        # 计算精确率、召回率和F1-score
        report = classification_report(all_labels, all_predictions)
        print("Classification Report:")
        print(report)
