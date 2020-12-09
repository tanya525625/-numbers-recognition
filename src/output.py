from tkinter import *
import tkinter as tk

import numpy as np
from PIL import ImageTk, Image


class App(tk.Tk):
    def __init__(self, pred , i):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=250)
        self.label = tk.Label(self, text=str(pred), font=("Helvetica", 48))
        self.canvas.pack()
        img = Image.fromarray(np.uint8(i)).convert('RGB')
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image = img)
        self.canvas.create_image(20, 20, anchor=NW, image=img)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.mainloop()
