"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train.py
 @DateTime: 2023/4/23 15:53
 @SoftWare: PyCharm
"""

from common.train import train
from common.test import test

if __name__ == '__main__':
    train(epochs=5, model_name='VGGNet', batch_size=32)
    test(model_name='VGGNet')
