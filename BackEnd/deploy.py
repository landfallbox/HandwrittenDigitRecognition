"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: convert_to_tensor.py
 @DateTime: 2023/3/6 15:42
 @SoftWare: PyCharm
"""
import base64
import os
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from flask import Flask, jsonify, request
from flask_cors import CORS

from common.args import device
from common.load_net import load_net
from common.load_transform import load_transform

import torch.nn.functional as F

app = Flask(__name__)
# 解决跨域问题
CORS(app, resources={r"/*": {"origins": "*"}})


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
    if model_name == 'LeNet5':
        source_length, target_length = 14, 28
    elif model_name == 'AlexNet':
        source_length, target_length = 112, 224
    else:
        raise ValueError('model_name must be LeNet5 or AlexNet')

    # 缩放图片，保持长宽比不变
    img.thumbnail((source_length, source_length))

    # 可视化缩放后的图片（测试）
    # desktop = "D:\\6429"
    # img.save('D:/6429/Pictures/after_resize.jpg')

    # 获取缩放后的宽度和高度
    source_width, source_height = img.size

    # 创建一个新的图像，背景为黑色，将原图像放在中间
    img_padded = Image.new('L', (target_length, target_length), 0)
    img_padded.paste(img, ((target_length - source_width) // 2, (target_length - source_height) // 2))

    return img_padded


# 图像预处理
def process_img(img, model):
    # 神经网络的输入张量，每个张量对应一个数字
    inputs = []

    # 颜色反转
    img = Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

    # 灰度化
    img = img.convert('L')

    # 二值化
    threshold = 100
    img = img.point(lambda x: 0 if x < threshold else 255)

    # desktop = os.path.expanduser("~/Desktop")
    # img.save(desktop + '/before_crop.jpg')

    # 图片转numpy数组，用于寻找轮廓
    img_np = np.array(img)

    # numpy数组转三通道图，用于绘制轮廓
    img_draw = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(image=img_np, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # 按轮廓的 x 坐标排序，以便从左到右逐个识别数字
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # 遍历轮廓
    for i, contour in enumerate(contours):
        # 在图片上绘制轮廓
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print(x, y, w, h)
        # cv2.imwrite(desktop + '/img_with_contour.jpg', img_draw)

        # 裁剪数字
        img_with_single_num = img.crop((x, y, x + w, y + h))
        # num.save(desktop + '/after_crop' + str(i) + '.jpg')

        # 调整大小
        img_with_single_num = resize_img(img_with_single_num, model)
        # num.save(desktop + '/after_padding' + str(i) + '.jpg')

        # 图片转为tensor
        transform = load_transform(model)
        img_with_single_num_tensor = transform(img_with_single_num).unsqueeze(0)
        # print(img_with_single_num_tensor.shape)

        # 添加tensor到列表中
        inputs.append(img_with_single_num_tensor)

    # 将绘制轮廓后的图片转为 base64 编码
    _, img_encoded = cv2.imencode('.jpg', img_draw)
    img_bytes = img_encoded.tobytes()
    img_with_contours_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return inputs, img_with_contours_base64


@app.route("/predict", methods=['POST'])
def predict():
    # 获取请求中的参数
    image_base64 = request.json.get('image')
    model = request.json.get('model')

    # 模型只支持 LeNet5, AlexNet 和 ResNet
    if model not in ['LeNet5', 'AlexNet', 'ResNet']:
        return jsonify(
            {
                'flag': 'fail',
                'results': {
                    'info': 'Wrong model name. Model must be LeNet5, AlexNet or Resnet. Check your request.',
                    'url': '',
                    'prediction(s)': ''
                }
            }
        )

    # 根据 url 获取图片
    img = get_img_from_url(image_base64)

    # 可视化图片（测试）
    save_path = 'D:/6429'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.convert('RGB').save(save_path + '/original.jpg')

    # 图像预处理，返回 tensor 列表和 base64 编码的图片
    # 每个 tensor 对应一个数字
    # 图片中给每个数字加上轮廓，方便查看
    inputs, img_base64 = process_img(img, model)

    # 加载模型
    net = load_net(model, device, torch.load('./model/' + model + '/model.pth'))
    net.eval()

    # 预测结果
    results = {}

    # 遍历处理后的图片 逐个预测结果
    for _, num_tensor in enumerate(inputs):
        num_tensor = num_tensor.to(device)

        # 神经网络识别手写数字
        prediction = net(num_tensor)
        print(prediction.data.tolist())

        # 获取预测结果和对应的概率
        predicted_label = torch.argmax(prediction, dim=1).item()
        predicted_prob = float(torch.max(F.softmax(prediction, dim=1)).cpu().detach().numpy())

        print('Prediction: ' + str(predicted_label))
        print('Probability: ' + str(predicted_prob))

        # 添加预测结果和概率到字典中
        if bool(results):
            results = {'prediction(s)': results.get('prediction(s)') + str(predicted_label),
                       'probability': results.get('probability') + predicted_prob}
        else:
            results = {'prediction(s)': str(predicted_label), 'probability': predicted_prob}

    # 添加预测成功的标志
    results = {
        'flag': 'success',
        'img': img_base64,
        'results': {
            'info': '',
            'prediction(s)': results.get('prediction(s)'),
            'probability': results.get('probability') / len(inputs)
        }
    }

    # json格式返回预测结果
    return jsonify(results)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
