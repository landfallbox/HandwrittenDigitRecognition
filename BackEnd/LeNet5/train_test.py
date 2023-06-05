"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train.py
 @DateTime: 2023/4/23 11:37
 @SoftWare: PyCharm
"""
from common.args import device
from common.test import test
from common.train import train

if __name__ == '__main__':
    train(epochs=5, model_name='LeNet5', device=device, batch_size=64, resume=False)
    # test(model_name='LeNet5', device=device)
