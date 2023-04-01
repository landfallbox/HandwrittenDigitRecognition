# 手写数字识别

## 项目描述

本项目是一个基于 CNN 的手写数字识别系统。用户在前端网页上的画板上书写数字，点击 “识别” 按钮，系统给出识别结果，即用户写的是数字几。

### 前端

前端页面使用WebStorm 2022.3.4 基于HTML 5、JS、CSS编写。

### 后端

后端使用 PyCharm 2022.2.3 基于 Python 3.8 编写。

借助 PyTorch 1.7.1、CUDA 11.0、CUDNN 8004 搭建、训练和测试神经网络。

使用 Flask 部署模型。

使用 Git 实现版本控制。

> 注意，PyTorch、CUDA、CUDNN 三者的版本取决于运行项目的电脑的 GPU，具体参考 NVIDIA 开发者网站，PyTorch 官网等。

## 项目结构

### 前端

```
FrontEnd
   │  index.html		主页面
   │      
   ├─css
   │      style.css		样式表
   └─js
   	draw.js		处理用户写数字，与后端之间的数据传输，将后端的预测结果展示在前端
```

### 后端

```
BackEnd
│  deploy.py         将模型部署到服务器上 
│  draw_loss.py      训练模型，测试模型，绘制全过程的损失变化       
│  MNIST2Img.py      将MNIST数据集的图片保存为jpg，标签保存为text  
│  model.pth         保存模型
│  net.py            定义神经网络模型，模型如何训练、如何测试
│  optimizer.pth     保存优化器内容
│               
└─data
  └─MNIST
      ├─processed
      │      
      └─raw             字节形式的MNIST数据集
```

# HandwrittenDigitRecognition

## Project description

The project is a handwritten digit recognition system based on CNN. When user write a number on the canvas on the web page and then click the "recognize" button, the system will give the prediction of what number did the user write. 

### Front end

The frontend page is written using WebStorm 2022.3.4 based on HTML 5, JS, and CSS.

### Back end

The backend is written using PyCharm 2022.2.3 based on Python 3.8.

Build, train, and test neural networks using PyTorch 1.7.1, CUDA 11.0, and CUDNN 8004.

Deploy the model using Flask.

Use Git for version control.

> Note that the versions of PyTorch, CUDA, and CUDNN depend on the GPU of the computer running the project. For details, refer to the NVIDIA developer website, PyTorch official website, and more.

## Project structure

### Front end

```
FrontEnd
│  index.html		the web page
│      
├─css
│      style.css    style sheet
└─js
		draw.js     enable users to write numbers on canvas, trasfer date to or receive data from the backend, show the prediction on the frontend
            

```



### Back end

```
BackEnd
│  deploy.py            deploy the network to the server
│  draw_loss.py         train, test the mmodule and record how the loss improve through the whole progress       
│  MNIST2Img.py         save the dataset MINST, img as jpg and lable as text  
│  model.pth            save the module
│  net.py               define the structure of the neural network, how to train and how to test
│  optimizer.pth        save optimizer
│               
└─data
  └─MNIST
      ├─processed
      │      
      └─raw             MNIST dataset in Bytes
```

> [用PyTorch实现MNIST手写数字识别(非常详细) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137571225)
>
> [快速上手_Flask中文网](https://flask.net.cn/quickstart.html#quickstart)
>
> [(55条消息) Python通过url获取图片的几种方法_写代码的胡歌的博客-CSDN博客](https://blog.csdn.net/qq_37124237/article/details/80931894)
>
> [(55条消息) (完全解决）Key already registered with the same priority: GroupSpatialSoftmax_音程的博客-CSDN博客](https://blog.csdn.net/qq_43391414/article/details/120096029)
>
> [(55条消息) 总结该问题解决方案：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized_peacefairy的博客-CSDN博客](https://blog.csdn.net/peacefairy/article/details/110528012)
>
> [(55条消息) 设置flask启动ip与端口_flask开放ip_bianlidou的博客-CSDN博客](https://blog.csdn.net/weixin_44936542/article/details/107343627)
>
> [ChatGPT](https://chat.openai.com/chat)