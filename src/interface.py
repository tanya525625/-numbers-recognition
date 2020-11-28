import win32gui
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=400, height=400, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def _canvas(self):
        print('  def _canvas(self):')
        print('self.cv.winfo_rootx() = ', self.canvas.winfo_rootx())
        print('self.cv.winfo_rooty() = ', self.canvas.winfo_rooty())
        print('self.cv.winfo_x() =', self.canvas.winfo_x())
        print('self.cv.winfo_y() =', self.canvas.winfo_y())
        print('self.cv.winfo_width() =', self.canvas.winfo_width())
        print('self.cv.winfo_height() =', self.canvas.winfo_height())
        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        box = (x, y, x1, y1)
        print('box = ', box)
        box_1 = (coord*1.5 for coord in box)
        return box_1

    def classify_handwriting(self):
        im = ImageGrab.grab().crop(self._canvas())
        im.show()

        # Prediction output
        # self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
