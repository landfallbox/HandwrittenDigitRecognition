"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train.py
 @DateTime: 2023/4/23 11:37
 @SoftWare: PyCharm
"""
from common.test import test
from common.train import train

if __name__ == '__main__':
    train(epochs=5, model_name='LeNet5')
    test(model_name='LeNet5')
