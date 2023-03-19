# 手写数字识别

## 项目描述

### 前端

前端页面使用HBuilder X基于HTML 5、JS、CSS编写。

### 后端

后端神经网络使用 PyCharm 2022.2.3 基于 Python 3.8 编写，借助 PyTorch 1.7.1、CUDA 11.0、CUDNN 8004 搭建、训练和测试网络，使用 Git 实现版本控制，使用TorchServe部署模型。注意，PyTorch、CUDA、CUDNN 三者的版本取决于运行项目的电脑的 GPU，具体参考 NVIDIA 开发者网站，PyTorch 官网等。

## 项目结构

### 前端

### 后端

```
HandwrittenDigitRecognition
│  Img2Tensor.py        把jpg转换成tensor
│  Mnist2Img.py         将MNIST数据集的图片保存为jpg，标签保存为text
│  model.pth            保存模型
│  net.py               定义神经网络模型
│  optimizer.pth        保存优化器内容
│  test_net.py          在测试集上测试已训练的模型的准确率
│  train_net.py         训练网络
│               
└─data
  └─MNIST
      ├─processed
      │      
      └─raw             字节形式的MNIST数据集
```

# HandwrittenDigitRecognition

## Project description

### Front end

### Back end

Back-end neural network is written using PyCharm 2022.2.3 based on Python 3.8, built, trained and tested with PyTorch 1.7.1, CUDA 11.0 and CUDNN 8004, and version control is implemented using Git. Note that the versions of PyTorch, CUDA and CUDNN depend on the GPU of the computer running the project. For details, please refer to the NVIDIA developer website and PyTorch official website.

## Project structure

### Front end

### Back end

```
HandwrittenDigitRecognition
│  Img2Tensor.py        tansfer jpg to tensor
│  Mnist2Img.py         save MNIST as jpg and labels as text
│  model.pth            where the network model is saved
│  net.py               define CNN
│  optimizer.pth        where the optimizer model is saved
│  test_net.py          test the accuracy of the tained network
│  train_net.py         train CNN
│               
└─data
  └─MNIST
      ├─processed
      │      
      └─raw             MNIST dataset in Bytes
```

> [用PyTorch实现MNIST手写数字识别(非常详细) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137571225)
>
> [(55条消息) js 使用HTML5 Websoket通信，发送和接收数据案例代码_js 发送 websocket 示例_程序媛zcx的博客-CSDN博客](https://blog.csdn.net/qq_40015157/article/details/114311028)
>
> [阿里云服务器搭建及项目部署过程---小白篇-阿里云开发者社区 (aliyun.com)
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
> [hbuilderx右边的预览工具栏不见了,怎么显示出来_百度知道 (baidu.com)](https://zhidao.baidu.com/question/1891174912089858628.html)