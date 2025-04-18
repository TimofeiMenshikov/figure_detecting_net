import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import threading
import queue
import time
import random

class RealTimePlot:
    def __init__(self, train_data_queue, valid_data_queue, shift):
        self.train_data_queue = train_data_queue
        self.valid_data_queue = valid_data_queue
        self.shift = shift
        
        # Инициализация графика
        self.fig, self.ax = plt.subplots()
        self.train_line, = self.ax.plot([], [], 'r-', label = "train")
        self.valid_line, = self.ax.plot([], [], 'b-', label = "valid")
        self.ax.legend()

        self.x_data = []
        self.train_y_data = []
        self.valid_y_data = []

        ax_button1 = plt.axes([0.9, 0.4, 0.1, 0.05])
        self.button1 = Button(ax_button1, "shift +")

        ax_button2 = plt.axes([0.9, 0.3, 0.1, 0.05])
        self.button2 = Button(ax_button2, "shift -")

        self.button1.on_clicked(self.increase_shift)  # Привязываем обработчик
        self.button2.on_clicked(self.decrease_shift)


    def get_data_from_queue(self, data_queue, y_data, x_data, fill_x_data = False):
        while not data_queue.empty():
            try:
                value = data_queue.get_nowait()
                print("value", value)
    
                
                y_data.append(value)

                if (fill_x_data):
                    x_data.append(len(y_data))

                data_queue.task_done()

            except queue.Empty:
                break

        return y_data, x_data
    

    def increase_shift(self, event): self.shift *= 2
    def decrease_shift(self, event): self.shift //= 2

        
    def update_plot(self, frame):
        """Обновление графика с сохранением истории"""
        # Сбор всех новых данных
        
        self.train_y_data, self.x_data = self.get_data_from_queue(self.train_data_queue, self.train_y_data, self.x_data, fill_x_data = False)
        self.valid_y_data, self.x_data = self.get_data_from_queue(self.valid_data_queue, self.valid_y_data, self.x_data, fill_x_data = True)


        min_len = min(len(self.x_data), len(self.train_y_data), len(self.valid_y_data))

        # Обновление данных графика
        self.train_line.set_data(self.x_data[:min_len], self.train_y_data[:min_len])
        self.valid_line.set_data(self.x_data[:min_len], self.valid_y_data[:min_len])


        if ((len(self.x_data) == 0) or (self.x_data[-1] < self.shift)): left = 0
        else:                                                           left = self.x_data[-1] - self.shift
        

        right = left + self.shift 
        
        self.ax.set_xlim(
            left  = left,
            right = right
        )

        self.ax.relim()                       # Пересчет лимитов на основе текущих данных
        self.ax.autoscale_view(scalex=False)  # Масштабируем только ось Y
        

        return self.train_line, self.valid_line


    def start(self):
        """Запуск анимации"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=20,            
        )

        plt.show()
