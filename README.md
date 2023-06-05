# 手写数字识别

## 项目描述

本项目是一个基于 CNN 的手写数字识别系统。用户在网页上书写数字，选择模型并点击预测按钮后网页展示预测结果和置信度。

## 系统架构

系统总体架构如图所示：

![image-20230605114137173](https://raw.githubusercontent.com/landfallbox/Pictures/master/image-20230605114137173.png)

### 前端

前端页面使用WebStorm 2022.3.4 基于HTML 5、JS、CSS、Bootstrap 编写。

### 后端

后端使用 PyCharm 2022.2.3 基于 Python 3.8 编写。主要包括两个部分：模型训练和模型部署。

模型训练时，基于MNIST数据集，借助 PyTorch 搭建、训练和测试神经网络。目前实现了 LeNet-5 和AlexNet。

模型部署时，借助 Flask 框架实现网页后端，实现请求监听和响应和图像预处理。

实现过程中使用的库如下所示：

```
Flask==2.2.3
Flask_Cors==3.0.10
matplotlib==3.3.4
numpy==1.23.5
opencv_python==4.4.0.44
Pillow==9.5.0
requests==2.28.2
scikit_image==0.19.3
scikit_learn==1.2.2
skimage==0.0
torch==1.7.1+cu110
torchvision==0.8.2+cu110
```

整个项目使用 Git 实现版本控制。

> 注意，PyTorch、CUDA、CUDNN 三者的版本取决于运行项目的电脑的 GPU，具体参考 NVIDIA 开发者网站，PyTorch 官网等。

## 代码结构

### 前端

```
D:.
│  index.html	页面
│  tree.txt		目录树
│  
├─.idea		配置文件
│          
├─css
│      style.css	样式表
│      
└─js
        script.js	JS文件，包括书写数字、前后端交互等
```

### 后端

```
D:.
│  .gitignore		Git配置文件
│  deploy.py		基于Flask实现网络后端，部署神经网络为Flask应用
│  MNIST2Img.py		MNIST数据集可视化
│  tree.txt
│  
├─.idea		
│          
├─AlexNet
│  │  net.py			AlexNet的网络模型
│  └─ train_test.py		调用train、test训练和测试AlexNet
│          
├─common
│  │  args.py				一些共用参数
│  │  load_net.py			加载网络
│  │  load_transform.py		加载图片转tensor的处理过程
│  │  test.py				模型测试函数
│  └─ train.py				模型训练函数
│          
├─data
│  └─MNIST		MNIST数据集
│              
├─LeNet5
│     net.py			LeNet-5的网络模型
│     train_test.py		调用train、test训练和测试LeNet-5
│          
└─model		loss最小的权重文件
        
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
>
> [(55条消息) cv2和PIL.Image之间的转换_pil image 转为cv2_绑个蝴蝶结的博客-CSDN博客](https://blog.csdn.net/qq_38153833/article/details/88060268)
>
> [(55条消息) 解决Pycharm等待工程update index时间太长_pycharm更新索引太慢_broad-sky的博客-CSDN博客](https://blog.csdn.net/qq_37164776/article/details/119281264)