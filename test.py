import tkinter as tk
from tkinter import ttk
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
        self.progressbar = ttk.Progressbar(self.master, variable=self.progress, maximum=100)
        self.progressbar.pack(pady=10)
        
        self.label_widget = tk.Label(self.master, textvariable=self.label)
        self.label_widget.pack(pady=10)

    def on_epoch_begin(self, epoch, logs=None):
        self.label.set(f"Epoch {epoch + 1} started")

    def on_epoch_end(self, epoch, logs=None):
        self.label.set(f"Epoch {epoch + 1} ended")
        self.progress.set((epoch + 1) / self.params['epochs'] * 100)

    def on_batch_end(self, batch, logs=None):
        self.label.set(f"Batch {batch + 1} completed")
        self.master.update()

# Create a simple model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(100,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create training data
X_train = np.random.rand(1000, 100)
y_train = np.random.randint(2, size=(1000, 1))

# Create the main window
root = tk.Tk()
root.title("Training Progress")
root.geometry("300x150")

# Initialize the progress bar and label
progress = TrainingProgress(root)

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[progress])

# Start the Tkinter event loop
root.mainloop()
