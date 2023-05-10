"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: load_transform.py
 @DateTime: 2023/4/28 23:26
 @SoftWare: PyCharm
"""
from torchvision import transforms


def load_transform(model_name):
    if model_name == 'CNN' or model_name == 'LeNet5':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif model_name == 'AlexNet':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        raise ValueError('model_name must be CNN, LeNet5 or AlexNet')

    return transform
