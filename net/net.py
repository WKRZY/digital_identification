# -*- coding : utf-8 -*-
# @Time      :2024-04-23 20:01
# @Author   : zy(子永)
# @ Software: Pycharm - windows
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    构造一个简单的卷积神经网络模型。

    该模型继承自nn.Module, 包含两个卷积层和两个全连接层。
    """

    def __init__(self):
        """
        初始化网络结构。
        """
        super(Net, self).__init__()  # 调用父类的初始化方法
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 第一层卷积，输入通道数为1，输出通道数为10，卷积核大小为5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 第二层卷积，输入通道数为10，输出通道数为20，卷积核大小为5x5
        self.conv2_drop = nn.Dropout2d()  # 用于卷积二层后的dropout操作，防止过拟合
        self.fc1 = nn.Linear(320, 50)  # 第一个全连接层，输入大小为320，输出大小为50
        self.fc2 = nn.Linear(50, 10)  # 第二个全连接层，输入大小为50，输出大小为10，通常用于分类任务的输出层

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
