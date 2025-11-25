import tkinter as tk
from tkinter import filedialog, messagebox
import os
import librosa
import soundfile as sf

def split_words(file_path, out_dir="splitted_words"):
    y, sr = librosa.load(file_path, sr=None)

    # Cắt theo khoảng lặng, cho nhạy hơn để tách từng từ
    intervals = librosa.effects.split(y, top_db=40, frame_length=512, hop_length=128)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = []
    for i, (start, end) in enumerate(intervals):
        segment = y[start:end]

        # bỏ noise ngắn <0.15s
        if (end - start) < sr * 0.15:
            continue

        out_file = os.path.join(out_dir, f"word_{i+1}.wav")
        sf.write(out_file, segment, sr)
        files.append(out_file)

    return files

def load_and_split():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        return
    try:
        out_files = split_words(file_path)
        messagebox.showinfo("Kết quả", f"Đã cắt thành {len(out_files)} file!\nLưu tại 'splitted_words/'")
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

root = tk.Tk()
root.title("WAV Splitter")

btn = tk.Button(root, text="Chọn WAV và Tách Từng Từ", command=load_and_split, width=40)
btn.pack(pady=20)

root.mainloop()
