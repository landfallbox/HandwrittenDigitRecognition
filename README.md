# 手写数字识别

## 项目描述

本项目使用 PyCharm 2022.2.3 基于 Python 3.9 编写，借助 PyTorch 1.7.1、CUDA 11.0、CUDNN 8004 搭建、训练和测试网络，使用 Git 实现版本控制。注意，PyTorch、CUDA、CUDNN 三者的版本取决于运行项目的电脑的 GPU，具体参考 NVIDIA 开发者网站，PyTorch 官网等。

## 项目结构

### 前端

### 后端

> 在初步构建项目时，主要参考了以下文章：[用PyTorch实现MNIST手写数字识别(非常详细) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137571225)

# HandwrittenDigitRecognition

## Project Structure

The IDE used to code this project is PyCharm 2022.2.3 based on Python 3.9 with PyTorch 1.7.1, CUDA 11.0, CUDNN 8004 used to build, train and test the network. Git is used to realize version control. What must be pointed out is version of PyTorch, CUDA and CUDNN is depended on the GPU of the computer on which the project will run. Check NVIDIA developer website, PyTorch official website for more specific info.

### net.py

Define what the CNN is.

### train_net.py

Define the way the network is trained.

### test_net.py

Define the way the network is tested.

### trans.py

Convert  image in JPG format into a tensor that can be recognized by the network.

### model.pth

This file is where network model and parameters are saved.

### optimizer.pth

This file is where optimizer is saved.

> During the preliminary construction of the project, the following article was mainly referred to: [用PyTorch实现MNIST手写数字识别(非常详细) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137571225) 