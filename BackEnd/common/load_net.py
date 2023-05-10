"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: load_net.py
 @DateTime: 2023/4/28 23:28
 @SoftWare: PyCharm
"""

from AlexNet.net import AlexNet
from CNN.net import CNN
from LeNet5.net import LeNet5


def load_net(model_name, device, model_info):
    if model_name == 'CNN':
        net = CNN().to(device)
    elif model_name == 'LeNet5':
        net = LeNet5().to(device)
    elif model_name == 'AlexNet':
        net = AlexNet().to(device)
    else:
        raise ValueError('model_name must be CNN, LeNet5 or AlexNet')

    if model_info is not None:
        net.load_state_dict(model_info['model_state_dict'])

    return net
