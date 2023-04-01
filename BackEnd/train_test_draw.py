"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: train_test_draw.py
 @DateTime: 2023/3/28 22:21
 @SoftWare: PyCharm
"""

import matplotlib.pyplot as plt

from BackEnd.net import epochs, train, test, train_counter, train_losses


def draw():
    # 画图
    fig = plt.figure()

    plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    # plt.plot(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')

    plt.title('Loss')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    plt.show()


for epoch in range(1, epochs + 1):
    train(epoch)
    test()
    # draw()
