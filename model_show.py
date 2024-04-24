# -*- coding : utf-8 -*-
# @Time      :2024-04-23 19:53
# @Author   : zy(子永)
# @ Software: Pycharm - windows
from tkinter import *
import tkinter as tk
import win32gui  # pip install pypiwin32
from PIL import ImageGrab
import numpy as np
import torch

from net.net import Net

model = Net()
model.load_state_dict(torch.load('model.pth'))


def predict_digit(img):
    """
    预测图像中的数字。

    参数:
    img - 输入的图像对象，需要先进行大小调整和灰度化处理。

    返回值:
    ped - 预测的数字类别。
    proba - 预测类别的概率。
    """

    # 调整图像大小为28x28像素
    img = img.resize((28, 28))
    # 将RGB图像转换为灰度图像
    img = img.convert('L')
    img = np.array(img)
    # 重新调整图像形状以适应模型输入，并进行归一化处理
    img = img.reshape((1, 1, 28, 28))
    img = 1 - img / 255.0  # 反转颜色，确保识别的图片是黑底白字

    # 将图像数据转换为张量
    img = torch.tensor(img, dtype=torch.float)
    # 进行预测
    res = model.forward(img)
    proba = float(max(torch.softmax(res, dim=1)[0]))  # 计算预测概率
    ped = int(torch.argmax(res))  # 获取预测类别
    return ped, proba


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        """
        根据鼠标事件，在画布上绘制一个黑色的圆点。

        参数:
        - self: 表示实例自身。
        - event: 鼠标事件对象，包含了事件发生时的坐标和其他信息。

        返回值:
        无。
        """
        self.x = event.x  # 获取事件的x坐标
        self.y = event.y  # 获取事件的y坐标
        r = 8  # 定义圆点的半径
        # 在画布上绘制一个黑色的圆点
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
