"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: convert_to_tensor.py
 @DateTime: 2023/3/6 15:42
 @SoftWare: PyCharm
"""
import base64
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from net import Net
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
# 启用调试模式
app.debug = True
# 解决跨域问题
CORS(app)


@app.route('/test', methods=['POST'])
def predict():
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

        # 灰度化
        img = img.convert('L')

        # 改变图片大小：28*28
        if img.size[0] != 28 or img.size[1] != 28:
            img = img.resize((28, 28))

        # 显示图片
        # img.show()

        # 转numpy数组
        img = np.array(img).astype(np.float32)

        # 数组从 28*28 转 1*1*28*28
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)

        # 转tensor
        img = torch.from_numpy(img)

        # 初始化网络，加载网络模型和参数
        network = Net()
        network.load_state_dict(torch.load('./model.pth'))

        # 测试模式
        network.eval()

        # 如果GPU可用则使用GPU训练和测试。否则，使用CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)
        img = img.to(device)

        # 预测手写数字
        predict = network(img).data.max(1, keepdim=True)[1].item()

        # 控制台输出预测结果
        print('predict:', predict)

        # 以json格式返回预测结果
        result = {'result': 'success', 'predict': predict}
        return jsonify(result)
    else:
        return jsonify({'result': 'fail', 'info': 'no url'})


if __name__ == '__main__':
    app.run()
