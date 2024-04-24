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
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape((1, 1, 28, 28))
    img = 1 - img / 255.0  # inverting the colors 识别的图片应该是黑底白字，若不是，则需要反转一下

    img = torch.tensor(img, dtype=torch.float)
    # predicting the class
    res = model.forward(img)
    proba = float(max(torch.softmax(res, dim=1)[0]))
    ped = int(torch.argmax(res))
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
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
