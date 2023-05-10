"""
 @Author: SheepDog
 @Email: landfallbox@gmail.com
 @FileName: args.py
 @DateTime: 2023/4/28 21:32
 @SoftWare: PyCharm
"""
import torch

# 优先使用GPU训练或测试
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 输出层的类别数
num_classes = 10

# 学习率
learning_rate = 0.01

# 动量因子
momentum = 0.9

# dropout概率
dropout = 0.5
