import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import os

# ==== CONFIG ====
DURATION = 2       # 2 giây ghi âm
SR = 16000         # sample rate
TEMP_FILE = "temp_record.wav"

# Mapping từ nhãn -> tiếng Việt có dấu (luôn viết hoa chữ đầu)
label_map_display = {
    "Bat": "Bật",
    "Tat": "Tắt",
    "Den": "Đèn",
    "Quat": "Quạt",
    "Ngu": "Ngủ",
    "Khach": "Khách",
    "Bep": "Bếp"
}

# Nhóm label
label_action = ["Bat", "Tat"]
label_device = ["Den", "Quat"]
label_room   = ["Ngu", "Khach", "Bep"]

# ==== GLOBAL ====
model = None
model_input_shape = None   # (batch, h, w, c)

# ==== FUNCTIONS ====
def load_model_file():
    global model, model_input_shape
    path = filedialog.askopenfilename(filetypes=[("Model H5", "*.h5")])
    if path:
        model = tf.keras.models.load_model(path)
        model_input_shape = model.input_shape  # vd: (None, 40, 64, 1)
        status_label.config(text=f"✅ Loaded model: {os.path.basename(path)} | input: {model_input_shape}")

def record_audio():
    status_label.config(text="🔴 Đang ghi âm 2 giây...")
    root.update()
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='int16')
    sd.wait()
    sf.write(TEMP_FILE, audio, SR)
    status_label.config(text="✅ Đã ghi xong")

def play_audio():
    if os.path.exists(TEMP_FILE):
        data, sr = sf.read(TEMP_FILE, dtype='float32')
        sd.play(data, sr)
        sd.wait()
    else:
        status_label.config(text="⚠ Chưa có file ghi âm")

def extract_features(file_path):
    """Chuẩn MFCC theo input_shape của model"""
    y, sr = librosa.load(file_path, sr=SR)

    if model_input_shape is None:
        raise ValueError("Model chưa load")

    target_shape = model_input_shape[1:]  # bỏ batch dim
    n_mfcc = target_shape[0]
    time_steps = target_shape[1]
    channels = target_shape[2] if len(target_shape) == 3 else 1

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < time_steps:
        mfcc = np.pad(mfcc, ((0,0),(0,time_steps-mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:,:time_steps]

    if len(target_shape) == 3:
        mfcc = mfcc.reshape((n_mfcc, time_steps, channels))
    else:
        mfcc = mfcc.reshape(target_shape)

    return mfcc

def predict_audio():
    if model is None:
        status_label.config(text="⚠ Chưa load model")
        return
    if not os.path.exists(TEMP_FILE):
        status_label.config(text="⚠ Chưa có file ghi âm")
        return

    try:
        feat = extract_features(TEMP_FILE)
        X = np.expand_dims(feat, axis=0)

        preds = model.predict(X)

        # Nếu output softmax 1 nhánh (n class)
        if isinstance(preds, list):
            preds = preds[0]  # lấy nhánh chính nếu có nhiều
        else:
            preds = preds[0]

        idx = np.argmax(preds)
        confidence = preds[idx]

        # === Map sang các nhóm ===
        labels = list(label_map_display.keys())
        detected_label = labels[idx] if idx < len(labels) else None

        # reset 3 textbox
        textbox_action.delete(0, tk.END)
        textbox_device.delete(0, tk.END)
        textbox_room.delete(0, tk.END)

        if detected_label in label_action:
            textbox_action.insert(0, label_map_display[detected_label])
        else:
            textbox_action.insert(0, "Null")

        if detected_label in label_device:
            textbox_device.insert(0, label_map_display[detected_label])
        else:
            textbox_device.insert(0, "Null")

        if detected_label in label_room:
            textbox_room.insert(0, label_map_display[detected_label])
        else:
            textbox_room.insert(0, "Null")

        status_label.config(text=f"✅ Detect: {detected_label} ({confidence:.2f})")

    except Exception as e:
        status_label.config(text=f"[ERROR] {e}")

# ==== GUI ====
root = tk.Tk()
root.title("Voice Command Detector")

# Buttons
btn_load = tk.Button(root, text="Load Model", command=load_model_file)
btn_load.grid(row=0, column=0, padx=5, pady=5)

btn_record = tk.Button(root, text="Record", command=record_audio)
btn_record.grid(row=0, column=1, padx=5, pady=5)

btn_play = tk.Button(root, text="Play", command=play_audio)
btn_play.grid(row=0, column=2, padx=5, pady=5)

btn_predict = tk.Button(root, text="Predict", command=predict_audio)
btn_predict.grid(row=0, column=3, padx=5, pady=5)

# Status
status_label = tk.Label(root, text="Chưa ghi âm")
status_label.grid(row=1, column=0, columnspan=4, pady=5)

# Textboxes
tk.Label(root, text="Hành động:").grid(row=2, column=0)
textbox_action = tk.Entry(root, width=20)
textbox_action.grid(row=2, column=1)

tk.Label(root, text="Thiết bị:").grid(row=3, column=0)
textbox_device = tk.Entry(root, width=20)
textbox_device.grid(row=3, column=1)

tk.Label(root, text="Phòng:").grid(row=4, column=0)
textbox_room = tk.Entry(root, width=20)
textbox_room.grid(row=4, column=1)

root.mainloop()
