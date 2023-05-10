"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train_test.py
 @DateTime: 2023/3/28 22:21
 @SoftWare: PyCharm
"""

from common.args import device
from common.test import test
from common.train import train

if __name__ == '__main__':
    train(epochs=5, model_name='CNN', device=device, batch_size=64, resume=False)
    test(model_name='CNN', device=device)
