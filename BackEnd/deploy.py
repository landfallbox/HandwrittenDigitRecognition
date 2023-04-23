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
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms

from CNN.net import device, CNN
from LeNet5.net import LeNet5
from flask import Flask, request, jsonify
from flask_cors import CORS

import os


app = Flask(__name__)
# 解决跨域问题
CORS(app)


def get_img_from_url(url):
    # 如果url是 base64 编码的图片数据
    if url.startswith('data:image/'):
        # 取出 base64 编码的数据部分
        img_data = url.split(',')[1]
        # 解码为二进制数据
        img_binary = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_binary))
    # 如果url是网络图片的地址
    else:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

    return img


# 图像预处理
def process_img(img, model_name):
    # 存放裁剪后的数字
    nums = []

    # 灰度化
    img = img.convert('L')

    # 二值化
    # threshold = 100
    # img = img.point(lambda x: 0 if x < threshold else 255)

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

        # 保存到桌面
        desktop = os.path.expanduser("~/Desktop")
        num.save(desktop + '/output.jpg')

        # print(num.size)

        # 转为numpy数组
        num = np.array(num).astype(np.float32)

        print(num.shape)

        # 数组从 28*28 转 1*1*28*28
        num = np.expand_dims(num, 0)
        num = np.expand_dims(num, 0)

        print(num.shape)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        num = transform(num)

        print(num.shape)

        # 将裁剪后的每个图片添加到列表中
        nums.append(num)

    return nums


def predict(model_name):
    # 读前端传来的json
    data = request.get_json()

    if 'url' in data:
        # 取图片的url
        url = data['url']
        img = get_img_from_url(url)
    else:
        return jsonify(
            {
                'flag': 'fail',
                'results': {
                    'info': 'no url',
                    'url': '',
                    'prediction(s)': ''
                }
            }
        )

    if model_name == 'CNN':
        # 图像预处理
        nums = process_img(img, model_name)

        # 加载CNN模型，测试模式
        net = CNN().to(device)
        net.load_state_dict(torch.load('./model/CNN/model.pth'))
        net.eval()
    elif model_name == 'LeNet5':
        # 图像预处理
        nums = process_img(img, model_name)

        # 加载LeNet5模型
        net = LeNet5().to(device)
        net.load_state_dict(torch.load('./model/LeNet5/model.pth'))
    elif model_name == 'VGGNet':
        # 图像预处理
        nums = process_img(img, model_name)

        # 加载VGGNet模型
        net = models.vgg16(pretrained=False).to(device)
        net.load_state_dict(torch.load('./model/VGGNet/model.pth'))
    else:
        # 模型名称不匹配
        return jsonify(
            {
                'flag': 'fail',
                'results': {
                    'info': 'wrong model name',
                    'url': '',
                    'prediction(s)': ''
                }
            }
        )

    # 存储预测结果
    results = {}

    for _, num in enumerate(nums):
        num = num.to(device)

        # 预测手写数字
        prediction = net(num).data.max(1, keepdim=True)[1].item()
        print(prediction)

        # 将预测结果添加到字典中
        if bool(results):
            results = {'prediction(s)': results.get('prediction(s)') + str(prediction)}
        else:
            results = {'prediction(s)': str(prediction)}

    # 添加预测成功的标志
    results = {
        'flag': 'success',
        'results': {
            'info': '',
            'url': url,
            'prediction(s)': results.get('prediction(s)')
        }
    }

    # json格式返回预测结果
    return jsonify(results)


@app.route('/predict_CNN', methods=['POST'])
def predict_CNN():
    return predict('CNN')


@app.route('/predict_LeNet5', methods=['POST'])
def predict_LeNet5():
    return predict('LeNet5')


@app.route('/predict_VGGNet', methods=['POST'])
def predict_VGGNet():
    return predict('VGGNet')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
