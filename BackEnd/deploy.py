"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: convert_to_tensor.py
 @DateTime: 2023/3/6 15:42
 @SoftWare: PyCharm
"""
import base64
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from torchvision.transforms import transforms

from net import network, device
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
# 启用调试模式
app.debug = True
# 解决跨域问题
CORS(app)


# 图像预处理
def process(img):
    # 存放裁剪后的数字
    nums = []

    # 灰度化
    img = img.convert('L')

    # 二值化
    threshold = 100
    img = img.point(lambda x: 0 if x < threshold else 255)

    # 转numpy数组
    img_np = np.array(img)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓并裁剪数字
    for _, contour in enumerate(contours):
        # 获取数字的矩形区域
        x, y, w, h = cv2.boundingRect(contour)

        # 将数字从原图中裁剪出来
        num = img.crop((x, y, x + w, y + h))

        # print(num.size)
        # num.show()

        # 改变图片大小：28*28
        if num.size[0] != 28 or num.size[1] != 28:
            num = num.resize((28, 28))

        # print(num.size)

        # 转为numpy数组
        num = np.array(num).astype(np.float32)

        print(num.shape)

        # 数组从 28*28 转 1*1*28*28
        num = np.expand_dims(num, 0)
        num = np.expand_dims(num, 0)

        print(num.shape)

        # 转为tensor
        num = torch.from_numpy(num)

        # 均值
        mean = 0.1307
        # 标准差
        std = 0.3081

        # 标准化 / 归一化
        transform = transforms.Normalize(mean=[mean], std=[std])
        num = transform(num)

        print(num.shape)

        # 将裁剪后的每个图片添加到列表中
        nums.append(num)

    return nums


@app.route('/test', methods=['POST'])
def predict_from_url():
    # 读取前端传来的数据
    data = request.get_json()

    if 'url' in data:
        # 取图片的url
        img_url = data['url']

        # 根据url读取图片
        if img_url.startswith('data:image/'):
            # 如果url是 base64 编码的图片数据
            img_data = img_url.split(',')[1]  # 取出 base64 编码的数据部分
            img_binary = base64.b64decode(img_data)  # 解码为二进制数据
            img = Image.open(BytesIO(img_binary))
        else:
            # 如果url是网络图片的地址
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))

        # img.show()

        # 图像预处理
        imgs = process(img)

        # 测试模式
        network.eval()

        # 存储预测结果
        results = {}

        for i, img in enumerate(imgs):
            img = img.to(device)

            # 预测手写数字
            prediction = network(img).data.max(1, keepdim=True)[1].item()

            if bool(results):
                # 将预测结果添加到字典中
                results = {'prediction': results.get('prediction') + str(prediction)}
            else:
                results = {'prediction': str(prediction)}

        # 添加预测成功的标志
        results = {'result': 'success', **results}

        # json格式返回预测结果
        return jsonify(results)
    else:
        return jsonify({'result': 'fail', 'info': 'no url'})


if __name__ == '__main__':
    app.run(debug=True)
