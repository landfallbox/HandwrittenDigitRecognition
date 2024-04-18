"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train_test.py
 @DateTime: 2023/4/28 21:11
 @SoftWare: PyCharm
"""
from common.args import device
from common.test import test
from common.train import train

if __name__ == '__main__':
    # train(epochs=10, model_name='AlexNet', device=device, batch_size=64, resume=False)
    test(model_name='AlexNet', device=device)
