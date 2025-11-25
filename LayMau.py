#!/usr/bin/env python3
"""
Audio Data Collector GUI (configurable recording time)
- Record configurable seconds at 16 kHz, 16-bit PCM
- Plot waveform
- Play and confirm recording before saving
- Sort into label folders
- Show two tables: existing labels (folders) and wav files

Dependencies:
    pip install sounddevice soundfile matplotlib numpy
"""
import os
import re
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = 'PCM_16'
OUTPUT_DIR = 'recordings'
FILENAME_TEMPLATE = '{label}_{num:04d}.wav'
RECORD_SECONDS = 1.0  # default recording time

os.makedirs(OUTPUT_DIR, exist_ok=True)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Audio Data Collector')
        self.geometry('1000x600')

        self.last_file = None
        self.temp_data = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.btn_record = ttk.Button(frm, text=f'Record', command=self.on_record)
        self.btn_record.pack(side=tk.LEFT, padx=4)

        self.btn_play = ttk.Button(frm, text='Play Last', command=self.on_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.btn_save = ttk.Button(frm, text='Save Recording', command=self.on_save)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.btn_sort = ttk.Button(frm, text='Sort by Label', command=self.on_sort)
        self.btn_sort.pack(side=tk.LEFT, padx=4)

        ttk.Label(frm, text='Saved folder:').pack(side=tk.LEFT, padx=(12,4))
        self.lbl_folder = ttk.Label(frm, text=OUTPUT_DIR)
        self.lbl_folder.pack(side=tk.LEFT)

        self.status_label = ttk.Label(self, text='', foreground='blue')
        self.status_label.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        plot_frm = ttk.Frame(self)
        plot_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig = Figure(figsize=(8,3))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Waveform (last recording)')
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')
        self.line, = self.ax.plot([], [])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frm)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_frm = ttk.Frame(self)
        bottom_frm.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: label folders
        lbl_frm = ttk.Frame(bottom_frm)
        lbl_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(lbl_frm, text='Existing Labels (folders):').pack(anchor='w')
        self.listbox_labels = tk.Listbox(lbl_frm)
        self.listbox_labels.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb1 = ttk.Scrollbar(lbl_frm, orient=tk.VERTICAL, command=self.listbox_labels.yview)
        sb1.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox_labels.config(yscrollcommand=sb1.set)

        # Right: wav files (only root)
        wav_frm = ttk.Frame(bottom_frm)
        wav_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(wav_frm, text='WAV Files in recordings/:').pack(anchor='w')
        self.listbox_files = tk.Listbox(wav_frm)
        self.listbox_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox_files.bind('<Double-Button-1>', self.on_list_double)
        sb2 = ttk.Scrollbar(wav_frm, orient=tk.VERTICAL, command=self.listbox_files.yview)
        sb2.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox_files.config(yscrollcommand=sb2.set)

        self._refresh_lists()

        # Entry for label input
        entry_frm = ttk.Frame(self)
        entry_frm.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)
        ttk.Label(entry_frm, text='Label:').pack(side=tk.LEFT)
        self.entry_label = ttk.Entry(entry_frm)
        self.entry_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Entry for record duration
        ttk.Label(entry_frm, text='Duration (s):').pack(side=tk.LEFT, padx=(12,4))
        self.entry_duration = ttk.Entry(entry_frm, width=6)
        self.entry_duration.insert(0, str(RECORD_SECONDS))
        self.entry_duration.pack(side=tk.LEFT)

    def on_record(self):
        def task():
            try:
                try:
                    duration = float(self.entry_duration.get())
                except ValueError:
                    duration = RECORD_SECONDS
                self.status_label.config(text=f'Đang ghi {duration:.1f} giây...')
                data = sd.rec(int(duration*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
                sd.wait()
                self.temp_data = data[:,0]
                self._plot_waveform(self.temp_data)
                # Play back for confirmation
                self.status_label.config(text='Phát lại để kiểm tra...')
                sd.play(self.temp_data.astype(np.float32) / 32768.0, SAMPLE_RATE)
                self.btn_record.config(text=f'Record {duration:.1f}s')
            except Exception as e:
                self.status_label.config(text=f'Lỗi: {e}')
        threading.Thread(target=task).start()

    def on_save(self):
        if self.temp_data is None:
            self.status_label.config(text='Không có dữ liệu để lưu.')
            return
        label = self.entry_label.get().strip()
        if not label:
            self.status_label.config(text='Label trống — không lưu.')
            return
        label = re.sub(r'[^A-Za-z0-9_\-]', '_', label)
        num = self._next_number_for_label(label)
        filename = FILENAME_TEMPLATE.format(label=label, num=num)
        filepath = os.path.join(OUTPUT_DIR, filename)
        sf.write(filepath, self.temp_data, SAMPLE_RATE, subtype=SUBTYPE)
        self.last_file = filepath
        self._refresh_lists()
        self.status_label.config(text=f'Đã lưu: {filepath}')

    def _next_number_for_label(self, label):
        pat = re.compile(re.escape(label) + r'_(\d+)\.wav$', re.IGNORECASE)
        nums = []
        for f in os.listdir(OUTPUT_DIR):
            m = pat.search(f)
            if m:
                try:
                    nums.append(int(m.group(1)))
                except:
                    pass
        return (max(nums)+1) if nums else 1

    def _plot_waveform(self, mono):
        self.ax.cla()
        self.ax.set_title('Waveform (last recording)')
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')
        x = np.arange(len(mono))
        self.ax.plot(x, mono)
        self.canvas.draw()

    def on_play(self):
        filepath = self.last_file
        if not filepath or not os.path.exists(filepath):
            self.status_label.config(text='Chưa có file để phát.')
            return
        data, sr = sf.read(filepath, dtype='int16')
        data_out = data.astype(np.float32) / 32768.0
        sd.play(data_out, sr)
        self.status_label.config(text=f'Đang phát: {os.path.basename(filepath)}')

    def on_list_double(self, event):
        sel_idx = self.listbox_files.curselection()
        if not sel_idx:
            return
        sel = self.listbox_files.get(sel_idx[0])
        filepath = os.path.join(OUTPUT_DIR, sel)
        if os.path.exists(filepath):
            data, sr = sf.read(filepath, dtype='int16')
            mono = data[:,0] if data.ndim==2 else data
            self.last_file = filepath
            self._plot_waveform(mono)
            self.status_label.config(text=f'Đã chọn: {filepath}')

    def _refresh_lists(self):
        # update files
        self.listbox_files.delete(0, tk.END)
        files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.wav')]
        files.sort()
        for f in files:
            self.listbox_files.insert(tk.END, f)
        # update labels (folders)
        self.listbox_labels.delete(0, tk.END)
        labels = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
        labels.sort()
        for l in labels:
            self.listbox_labels.insert(tk.END, l)

    def on_sort(self):
        files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.wav')]
        moved = 0
        for f in files:
            m = re.match(r'([^_]+)_(\d+)\.wav$', f, re.IGNORECASE)
            if m:
                label = m.group(1)
                src = os.path.join(OUTPUT_DIR, f)
                dest_dir = os.path.join(OUTPUT_DIR, label)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, f)
                try:
                    os.replace(src, dest)
                    moved += 1
                except Exception as e:
                    print('Move error', e)
        self._refresh_lists()
        self.status_label.config(text=f'Đã di chuyển {moved} file.')

if __name__ == '__main__':
    app = App()
    app.mainloop()
