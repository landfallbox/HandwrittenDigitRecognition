"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: net.py
 @DateTime: 2023/3/3 16:24
 @SoftWare: PyCharm
"""

from torch import nn
from common.args import num_classes, dropout


# CNN
class CNN(nn.Module):
    def __init__(self, num_classes=num_classes, dropout=dropout):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return x
