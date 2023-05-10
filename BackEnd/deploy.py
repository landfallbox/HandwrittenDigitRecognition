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

from flask import Flask, request, jsonify
from flask_cors import CORS

import os

from common.args import device
from common.load_net import load_net
from common.load_transform import load_transform

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


# 调整图片大小
def resize_img(img, model_name):
    # 根据模型的不同，设置不同的图片缩放后的大小和填充后的大小
    if model_name == 'CNN' or model_name == 'LeNet5':
        source_length, target_length = 14, 28
    elif model_name == 'AlexNet':
        source_length, target_length = 112, 224
    else:
        raise ValueError('model_name must be CNN, LeNet5 or AlexNet')

    # 缩放图片，保持长宽比不变
    img.thumbnail((source_length, source_length))

    # 可视化缩放后的图片（测试）
    desktop = os.path.expanduser("~/Desktop")
    img.save(desktop + '/after_resize.jpg')

    # 获取缩放后的宽度和高度
    source_width, source_height = img.size

    # 创建一个新的图像，背景为黑色，将原图像放在中间
    img_padded = Image.new('L', (target_length, target_length), 0)
    img_padded.paste(img, ((target_length - source_width) // 2, (target_length - source_height) // 2))

    return img_padded


# 图像预处理
def process_img(img, model_name):
    # 存放裁剪后的数字
    nums = []

    # 灰度化
    img = img.convert('L')

    # 二值化
    threshold = 100
    img = img.point(lambda x: 0 if x < threshold else 255)

    desktop = os.path.expanduser("~/Desktop")
    img.save(desktop + '/before_crop.jpg')

    # 转numpy数组
    img_np = np.array(img)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按轮廓的 x 坐标排序，以便从左到右逐个识别数字
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # 遍历轮廓并裁剪数字
    for i, contour in enumerate(contours):
        # 获取数字的矩形区域
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)

        # 裁剪数字
        num = img.crop((x, y, x + w, y + h))
        num.save(desktop + '/after_crop' + str(i) + '.jpg')

        # 调整大小
        num = resize_img(num, model_name)
        num.save(desktop + '/after_padding' + str(i) + '.jpg')

        # 转为tensor
        transform = load_transform(model_name)
        num = transform(num).unsqueeze(0)
        print(num.shape)

        # 将裁剪后的每个图片添加到列表中
        nums.append(num)

    return nums


def predict(model_name):
    # 读前端传来的json
    data = request.get_json()

    # 取url，如果没有url则返回错误信息。根据url获取图片
    if 'url' in data:
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

    # 图像预处理
    nums = process_img(img, model_name)

    # 加载模型
    net = load_net(model_name, device, torch.load('./model/' + model_name + '/model.pth'))
    net.eval()

    # 模型名称不匹配
    if not bool(net):
        return jsonify(
            {
                'flag': 'fail',
                'results': {
                    'info': 'Wrong model name. Model must be CNN, LeNet5, AlexNet or VGGNet. Check your request.',
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
        prediction = net(num)
        print(prediction.data.tolist())
        prediction = torch.argmax(prediction, dim=1).item()
        print('Prediction: ' + str(prediction))

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


@app.route('/predict/CNN', methods=['POST'])
def predict_CNN():
    return predict('CNN')


@app.route('/predict/LeNet5', methods=['POST'])
def predict_LeNet5():
    return predict('LeNet5')


@app.route('/predict/AlexNet', methods=['POST'])
def predict_AlexNet():
    return predict('AlexNet')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
