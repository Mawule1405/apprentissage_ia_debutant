import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class TrainingProgress(Callback):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.progress = tk.DoubleVar()
        self.label = tk.StringVar()
        self.create_widgets()
    
    def create_widgets(self):
        self.progressbar = ttk.Progressbar(self.master,length=1000, variable=self.progress, maximum=100)
        self.progressbar.pack(pady=5, padx=7)
        
        self.label_widget = ctk.CTkLabel(self.master, textvariable=self.label)
        self.label_widget.pack(pady=2)

    def on_epoch_begin(self, epoch, logs=None):
        self.label.set(f"Epoch {epoch + 1} started")

    def on_epoch_end(self, epoch, logs=None):
        self.label.set(f"Epoch {epoch + 1} ended")
        self.progress.set((epoch + 1) / self.params['epochs'] * 100)

    def on_batch_end(self, batch, logs=None):
        self.label.set(f"Batch {batch + 1} completed")
        self.master.update()

    
    def reset_progress(self):
        self.progress.set(0)
        self.label.set("")


