# 手写数字识别

本仓库是本科毕业设计：用 PyTorch 实现 MNIST 手写数字识别。

## 项目结构

本项目使用 PyCharm 2022.2.3 基于 Python 3.9 编写，借助 PyTorch 1.7.1、CUDA 11.0、CUDNN 8004 搭建、训练和测试网络，使用 Git 实现版本控制。注意，PyTorch、CUDA、CUDNN 三者的版本取决于运行项目的电脑的 GPU，具体参考 NVIDIA 开发者网站，PyTorch 官网等。

### main.py

由于手写数字识别比较简单，所以项目所有代码全部写在 main.py 中。其中既包括加载数据集，定义网络模型，定义训练和测试函数，还包括构建网络，加载模型和参数至 GPU，训练和测试网络。

### model.pth

保存网络模型和参数等数据。

### optimizer.pth

保存优化器的数据。

> 在初步构建项目时，主要参考了以下文章：[用PyTorch实现MNIST手写数字识别(非常详细) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137571225)

# HandwrittenDigitRecognition

This repository is a graduation project for undergraduates: using PyTorch to realize MNIST handwritten digit recognition.

## Project Structure

The IDE used to code this project is PyCharm 2022.2.3 based on Python 3.9 with PyTorch 1.7.1, CUDA 11.0, CUDNN 8004 used to build, train and test the network. Git is used to realize version control. What must be pointed out is version of PyTorch, CUDA and CUDNN is depended on the GPU of the computer on which the project will run. Check NVIDIA developer website, PyTorch official website for more specific info.

### main.py

Due to handwritten digit recognition is relatively simple, all codes are putted in main.py from loading data sets, defining network models,  defining training and testing functions to building network, loading model and parameters to GPU, training and testing network.

### model.pth

This file is where network model and parameters are saved.

### optimizer.pth

This file is where optimizer is saved.

> During the preliminary construction of the project, the following article was mainly referred to: [用PyTorch实现MNIST手写数字识别(非常详细) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137571225) 