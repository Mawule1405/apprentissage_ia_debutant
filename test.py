import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
import numpy as np

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Monitor")
        
        self.create_widgets()
        
    def create_widgets(self):
        self.start_button = tk.Button(self.root, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=10)
        
        self.figure, self.ax = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()

    def start_training(self):
        self.start_button.config(state=tk.DISABLED)
        self.train_model()

    def train_model(self):
        # Load data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128,
                  verbose=0, callbacks=[self.LossHistory(self)])

    class LossHistory(keras.callbacks.Callback):
        def __init__(self, gui):
            self.gui = gui

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.gui.history['loss'].append(logs.get('loss'))
            self.gui.history['accuracy'].append(logs.get('accuracy'))
            self.gui.history['val_loss'].append(logs.get('val_loss'))
            self.gui.history['val_accuracy'].append(logs.get('val_accuracy'))
            self.gui.update_plot()

    def update_plot(self):
        self.ax[0].clear()
        self.ax[1].clear()
        
        self.ax[0].plot(self.history['loss'], label='Train Loss')
        self.ax[0].plot(self.history['val_loss'], label='Val Loss')
        self.ax[0].set_title('Loss')
        self.ax[0].legend()
        
        self.ax[1].plot(self.history['accuracy'], label='Train Accuracy')
        self.ax[1].plot(self.history['val_accuracy'], label='Val Accuracy')
        self.ax[1].set_title('Accuracy')
        self.ax[1].legend()
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    gui = TrainingGUI(root)
    root.mainloop()
