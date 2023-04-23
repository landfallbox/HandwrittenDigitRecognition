"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: try.py
 @DateTime: 2023/4/20 20:56
 @SoftWare: PyCharm
"""
import torch
from PIL import Image
import numpy as np

from torchvision.transforms import transforms

from net import Net, device, ResNet, Basicblock

# img = Image.open("C:/Users/64297/Desktop/output.jpg")
img = Image.open('../data/test/2.jpg')
# img.show()
# 转为numpy数组
num = np.array(img).astype(np.float32)
# print(img_np.shape)

# 数组从 28*28 转 1*1*28*28
num = np.expand_dims(num, 0)
num = np.expand_dims(num, 0)

num = torch.from_numpy(num)

# 均值
mean = 0.1307
# 标准差
std = 0.3081
# 标准化 / 归一化
transform = transforms.Normalize(mean=[mean], std=[std])
num = transform(num)

num = num.to(device)

network = Net().to(device)
# network = LeNet5(Basicblock, [1, 1, 1, 1], 10).to(device)
# 加载网络模型和参数
network.load_state_dict(torch.load('./model/model.pth'))
network.eval()

prediction = network(num).data.max(1, keepdim=True)[1].item()
print(prediction)
