"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: convert_to_img.py
 @DateTime: 2023/3/6 14:54
 @SoftWare: PyCharm
"""

import os
from skimage import io
import torchvision.datasets.mnist as mnist

root = "./data/MNIST/raw"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

print("training set :", train_set[0].size())
print("test set :", test_set[0].size())


def convert(file, data_path, data_set):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for i, (img, label) in enumerate(zip(data_set[0], data_set[1])):
        img_path = data_path + str(i) + '.jpg'
        io.imsave(img_path, img.numpy())
        file.write(img_path + ' ' + str(label) + '\n')

    file.close()


def convert_to_img(train=True):
    if train:
        file = open('data/train.txt', 'w')
        data_path = 'data/train/'

        convert(file, data_path, train_set)
    else:
        file = open('data/test.txt', 'w')
        data_path = 'data/test/'

        convert(file, data_path, test_set)


convert_to_img(True)  # 转换训练集
convert_to_img(False)  # 转换测试集
