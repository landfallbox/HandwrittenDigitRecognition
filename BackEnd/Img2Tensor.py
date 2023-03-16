"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: convert_to_tensor.py
 @DateTime: 2023/3/6 15:42
 @SoftWare: PyCharm
"""

import numpy as np
import torch
from PIL import Image
from net import Net


# noinspection PyShadowingNames
def convert_to_tensor(img_path):
    # 读图片，灰度化
    img = Image.open(img_path).convert('L')
    # img.show()

    # 改变图片大小：28*28
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))
    # img.show()

    # img = img.rotate(90)

    # 转numpy数组
    img = np.array(img).astype(np.float32)

    # 图片灰度后，是白底黑字，MNIST恰好相反
    temp = np.ones([28, 28], dtype=np.float32)
    for i in range(28):
        for j in range(28):
            temp[i, j] = 1 - (img[i, j] / 255 - 0.1307) / 0.3081
    img = temp

    # 数组从 28*28 转 1*1*28*28
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    # 转tensor
    img = torch.from_numpy(img)

    return img


# 初始化网络，加载网络模型和参数
network = Net()
network.load_state_dict(torch.load('./model.pth'))

# 测试模式
network.eval()

# 如果GPU可用则使用GPU训练和测试。否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)

img = convert_to_tensor('./test/1.jpg').to(device)
predict = network(img).data.max(1, keepdim=True)[1]
print(predict)
