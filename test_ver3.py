#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf

SAMPLE_RATE = 16000
N_MFCC = 40   # số Mel filter (giữ tên cũ N_MFCC để tương thích)
MAX_LEN = 64
CHANNELS = 1
RECORD_FILE = "temp_record.wav"

LABEL_GROUPS = ["Action", "Device", "Room"]

class VoiceTestGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Test - MFE")
        self.geometry("600x450")

        self.models = {name: None for name in LABEL_GROUPS}
        self.label_maps = {name: {} for name in LABEL_GROUPS}
        self.result_vars = {name: tk.StringVar(value="") for name in LABEL_GROUPS}

        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(pady=10)

        ttk.Button(top_frame, text="Record", command=self._record).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Play", command=self._play).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Predict", command=self._predict).pack(side=tk.LEFT, padx=5)

        for name in LABEL_GROUPS:
            frame = ttk.Frame(self)
            frame.pack(fill=tk.X, padx=10, pady=5)
            ttk.Label(frame, text=name + ":").pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="Load Model", command=lambda n=name: self._load_model(n)).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=self.result_vars[name], width=40).pack(side=tk.LEFT, padx=5)

        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(status_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT)

    def _record(self):
        duration = 2  # seconds
        self._update_status(f"Recording for {duration} seconds...")
        self.update_idletasks()
        recording = sd.rec(int(duration*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        sf.write(RECORD_FILE, recording, SAMPLE_RATE)
        self._update_status(f"Recording saved to {RECORD_FILE}")

    def _play(self):
        if not os.path.exists(RECORD_FILE):
            self._update_status("No recording found")
            return
        data, sr = sf.read(RECORD_FILE)
        self._update_status("Playing recording...")
        self.update_idletasks()
        sd.play(data, sr)
        sd.wait()
        self._update_status("Playback finished")

    def _load_model(self, group_name):
        path = filedialog.askopenfilename(filetypes=[("H5 files","*.h5")])
        if not path:
            self._update_status(f"Load model canceled for {group_name}")
            return
        try:
            model = tf.keras.models.load_model(path)
            self.models[group_name] = model
            # load label map
            label_file = path + "_labels.json"
            if os.path.exists(label_file):
                import json
                with open(label_file, "r") as f:
                    self.label_maps[group_name] = json.load(f)
            self._update_status(f"{group_name} model loaded: {os.path.basename(path)}")
        except Exception as e:
            self._update_status(f"Error loading {group_name} model: {e}")

    def _predict(self):
        if not os.path.exists(RECORD_FILE):
            self._update_status("No recording to predict")
            return

        # Reset all result textboxes
        for group in LABEL_GROUPS:
            self.result_vars[group].set("")

        y, sr = librosa.load(RECORD_FILE, sr=SAMPLE_RATE, mono=True)
        # --- MFE (log-Mel Energy) ---
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MFCC,
            fmin=20,
            fmax=4000,
            power=2.0
        )
        mfe = librosa.power_to_db(S, ref=np.max)

        if mfe.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfe.shape[1]
            mfe = np.pad(mfe, ((0,0),(0,pad_width)), mode="constant")
        else:
            mfe = mfe[:, :MAX_LEN]

        mfe = mfe[np.newaxis, ..., np.newaxis]  # shape (1,40,64,1)

        for group in LABEL_GROUPS:
            model = self.models[group]
            if model is None:
                self.result_vars[group].set("No model loaded")
                continue
            pred = model.predict(mfe, verbose=0)
            idx = np.argmax(pred)
            label = self.label_maps[group].get(str(idx), f"Label {idx}")
            self.result_vars[group].set(label)

        self._update_status("Prediction done")

    def _update_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

if __name__ == "__main__":
    app = VoiceTestGUI()
    app.mainloop()
