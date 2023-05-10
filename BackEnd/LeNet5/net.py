"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: net.py
 @DateTime: 2023/4/23 11:33
 @SoftWare: PyCharm
"""
from torch import nn
from common.args import num_classes


# LeNet5
class LeNet5(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x
