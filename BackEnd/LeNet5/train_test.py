"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train.py
 @DateTime: 2023/4/23 11:37
 @SoftWare: PyCharm
"""
from common.train import train
from common.test import test

if __name__ == '__main__':
    train(epochs=5, model_name='LeNet5')
    test(model_name='LeNet5')
