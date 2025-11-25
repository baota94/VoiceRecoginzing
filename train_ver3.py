#!/usr/bin/env python3
"""
Voice Trainer with TensorFlow + Combo Box + MFE (Mel Filterbank Energy)
- Dataset: dataset/<label>/*.wav
- Combo Box: Action / Device / Room
- Train model chỉ với các nhãn tương ứng
- Save model automatically as <Model>_<num>.h5
"""

import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
import tensorflow as tf

SAMPLE_RATE = 16000
N_MFCC = 40        # số filter của Mel
MAX_LEN = 64       # số frame MFE tối đa (pad/truncate)
CHANNELS = 1

LABEL_GROUPS = {
    "Action": ["Bat", "Tat"],
    "Device": ["Den", "Quat"],
    "Room": ["Ngu", "Khach", "Bep"]
}

class VoiceTrainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Trainer (TensorFlow - MFE)")
        self.geometry("1000x700")

        self.dataset_dir = tk.StringVar(value="dataset")
        self.epochs = tk.IntVar(value=20)
        self.batch_size = tk.IntVar(value=32)
        self.test_split = tk.DoubleVar(value=0.2)
        self.model_type = tk.StringVar(value="Action")

        self.model = None
        self.history = None
        self.labels = []

        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        # Dataset folder
        ttk.Label(ctrl, text="Dataset dir:").pack(side=tk.LEFT)
        ttk.Entry(ctrl, textvariable=self.dataset_dir, width=30).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Browse", command=self._browse_dataset).pack(side=tk.LEFT, padx=4)

        # Combo box: Action / Device / Room
        ttk.Label(ctrl, text="Type:").pack(side=tk.LEFT, padx=(20,0))
        combo = ttk.Combobox(ctrl, textvariable=self.model_type, values=list(LABEL_GROUPS.keys()), state="readonly", width=10)
        combo.pack(side=tk.LEFT)

        # Epochs / Batch / Test split
        ttk.Label(ctrl, text="Epochs:").pack(side=tk.LEFT, padx=(20,0))
        ttk.Entry(ctrl, textvariable=self.epochs, width=5).pack(side=tk.LEFT)
        ttk.Label(ctrl, text="Batch:").pack(side=tk.LEFT, padx=(20,0))
        ttk.Entry(ctrl, textvariable=self.batch_size, width=5).pack(side=tk.LEFT)
        ttk.Label(ctrl, text="Test split:").pack(side=tk.LEFT, padx=(20,0))
        ttk.Entry(ctrl, textvariable=self.test_split, width=5).pack(side=tk.LEFT)

        # Buttons
        ttk.Button(ctrl, text="Train", command=self._train_thread).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text="Save Model", command=self._save_model).pack(side=tk.LEFT, padx=10)

        # Chart
        fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(9,4))
        self.ax_loss.set_title("Loss")
        self.ax_acc.set_title("Accuracy")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log
        self.txt_log = tk.Text(self, height=10)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _log(self, *args):
        s = " ".join(str(a) for a in args) + "\n"
        self.txt_log.insert(tk.END, s)
        self.txt_log.see(tk.END)

    def _browse_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_dir.set(path)

    def _load_dataset(self, base_dir):
        X, Y = [], []
        selected_labels = LABEL_GROUPS[self.model_type.get()]
        self.labels = selected_labels
        self._log("Selected labels:", selected_labels)

        for idx, label in enumerate(selected_labels):
            folder = os.path.join(base_dir, label)
            if not os.path.exists(folder):
                self._log("Folder not found:", folder)
                continue
            files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
            for f in files:
                path = os.path.join(folder, f)
                try:
                    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                    # Compute Mel Filterbank Energy (MFE)
                    S = librosa.feature.melspectrogram(
                        y=y,
                        sr=sr,
                        n_mels=N_MFCC,
                        fmin=20,
                        fmax=4000,
                        power=2.0  # energy
                    )
                    mfe = librosa.power_to_db(S, ref=np.max)

                    # pad/truncate về MAX_LEN frame
                    if mfe.shape[1] < MAX_LEN:
                        pad_width = MAX_LEN - mfe.shape[1]
                        mfe = np.pad(mfe, ((0,0),(0,pad_width)), mode="constant")
                    else:
                        mfe = mfe[:, :MAX_LEN]

                    mfe = mfe[..., np.newaxis]  # add channel dimension
                    X.append(mfe)
                    Y.append(idx)
                except Exception as e:
                    self._log("Error loading", path, e)
        X = np.array(X)
        Y = np.array(Y)
        self._log("Loaded dataset:", X.shape, "labels:", len(selected_labels))
        return X, Y

    def _build_model(self, input_shape, num_classes):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def _train_thread(self):
        threading.Thread(target=self._train).start()

    def _train(self):
        X, Y = self._load_dataset(self.dataset_dir.get())
        if len(X) == 0:
            self._log("Dataset is empty")
            return
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=float(self.test_split.get()), random_state=42, stratify=Y)

        self.model = self._build_model((N_MFCC, MAX_LEN, CHANNELS), len(self.labels))

        self._log("Start training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=int(self.epochs.get()),
            batch_size=int(self.batch_size.get()),
            validation_data=(X_test, y_test),
            verbose=0
        )
        self.history = history

        self._update_chart(history)
        self._log("Training done. Val acc:", history.history['val_accuracy'][-1])
        # Auto-save model
        self._auto_save_model()

    def _update_chart(self, history):
        self.ax_loss.cla()
        self.ax_acc.cla()
        self.ax_loss.plot(history.history['loss'], label="train")
        self.ax_loss.plot(history.history['val_loss'], label="val")
        self.ax_loss.set_title("Loss")
        self.ax_loss.legend()
        self.ax_acc.plot(history.history['accuracy'], label="train")
        self.ax_acc.plot(history.history['val_accuracy'], label="val")
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.legend()
        self.canvas.draw()

    def _auto_save_model(self):
        base_name = self.model_type.get()
        i = 1
        while os.path.exists(f"{base_name}_{i}.h5"):
            i += 1
        path = f"{base_name}_{i}.h5"
        self.model.save(path)
        labelmap = {i: l for i,l in enumerate(self.labels)}
        with open(path + "_labels.json", "w") as f:
            json.dump(labelmap, f)
        self._log("Model auto-saved to", path)

    def _save_model(self):
        if self.model is None:
            self._log("No model trained yet")
            return
        path = filedialog.asksaveasfilename(defaultextension=".h5")
        if not path:
            return
        self.model.save(path)
        labelmap = {i: l for i,l in enumerate(self.labels)}
        with open(path + "_labels.json", "w") as f:
            json.dump(labelmap, f)
        self._log("Model saved to", path)

if __name__ == "__main__":
    app = VoiceTrainer()
    app.mainloop()
