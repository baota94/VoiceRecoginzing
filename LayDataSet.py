#!/usr/bin/env python3
"""
Dataset Builder (standalone)
- Scans a base recordings/ directory for subfolders (labels)
- For each WAV file in subfolders, applies simple VAD using librosa.effects.split
  (based on top_db) to find non-silent intervals
- Options let you choose to keep the longest interval, concatenate intervals, or skip
  files shorter than a minimum duration
- Saves processed WAV files into dataset/<label>/ with 16 kHz, 16-bit PCM, mono
- Preview pane to listen to original and processed audio before saving
- Shows waveform chart for original and processed audio with highlights for kept intervals
- File selection dialog to manually choose a file for inspection

Dependencies:
    pip install numpy soundfile librosa sounddevice matplotlib

Usage:
    python dataset_builder.py
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Config
SAMPLE_RATE = 16000
SUBTYPE = 'PCM_16'
RECORDINGS_DIR = 'recordings'
OUTPUT_DIR = 'dataset'

os.makedirs(OUTPUT_DIR, exist_ok=True)

class DatasetBuilder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Dataset Builder')
        self.geometry('1200x750')

        self.top_db = tk.DoubleVar(value=25.0)
        self.min_duration = tk.DoubleVar(value=0.2)
        self.method = tk.StringVar(value='longest')  # 'longest' | 'concat'

        self.current_file = None
        self.current_y = None
        self.processed_y = None
        self.intervals = []

        self._build_ui()
        self._refresh_label_list()

    def _build_ui(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(ctrl, text='Recordings folder:').pack(side=tk.LEFT)
        self.lbl_rec = ttk.Label(ctrl, text=RECORDINGS_DIR)
        self.lbl_rec.pack(side=tk.LEFT, padx=(4,12))

        ttk.Label(ctrl, text='top_db:').pack(side=tk.LEFT)
        ttk.Entry(ctrl, textvariable=self.top_db, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text='min dur (s):').pack(side=tk.LEFT)
        ttk.Entry(ctrl, textvariable=self.min_duration, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text='method:').pack(side=tk.LEFT)
        ttk.Combobox(ctrl, textvariable=self.method, values=['longest', 'concat'], width=8).pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text='Select File...', command=self._select_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text='Refresh', command=self._refresh_label_list).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text='Process Selected Label', command=self._process_selected_label).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text='Process All Labels', command=self._process_all_labels).pack(side=tk.LEFT, padx=6)

        # Left: labels and files
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Label(left, text='Labels (subfolders):').pack(anchor='w')
        self.lb_labels = tk.Listbox(left, height=12)
        self.lb_labels.pack(fill=tk.BOTH, expand=True)
        self.lb_labels.bind('<<ListboxSelect>>', self._on_label_select)

        ttk.Label(left, text='Files in label:').pack(anchor='w', pady=(6,0))
        self.lb_files = tk.Listbox(left, height=12)
        self.lb_files.pack(fill=tk.BOTH, expand=True)
        self.lb_files.bind('<<ListboxSelect>>', self._on_file_select)

        # Right: preview + chart + logs
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        preview_ctrl = ttk.Frame(right)
        preview_ctrl.pack(fill=tk.X)
        ttk.Button(preview_ctrl, text='Play Original', command=self.play_original).pack(side=tk.LEFT, padx=4)
        ttk.Button(preview_ctrl, text='Preview Processed', command=self.play_processed).pack(side=tk.LEFT, padx=4)
        ttk.Button(preview_ctrl, text='Save Processed (single)', command=self.save_processed_single).pack(side=tk.LEFT, padx=4)

        # Chart area
        fig = Figure(figsize=(6,4))
        self.ax_orig = fig.add_subplot(211)
        self.ax_proc = fig.add_subplot(212)
        self.ax_orig.set_title('Original Waveform (with intervals)')
        self.ax_proc.set_title('Processed Waveform')
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=6)

        self.txt_log = tk.Text(right, height=10)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _log(self, *args):
        s = ' '.join(str(a) for a in args) + '\n'
        self.txt_log.insert(tk.END, s)
        self.txt_log.see(tk.END)

    def _refresh_label_list(self):
        self.lb_labels.delete(0, tk.END)
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
        labels = [d for d in os.listdir(RECORDINGS_DIR) if os.path.isdir(os.path.join(RECORDINGS_DIR, d))]
        labels.sort()
        for l in labels:
            self.lb_labels.insert(tk.END, l)
        self._log('Found labels:', len(labels))

    def _select_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files","*.wav")], initialdir=RECORDINGS_DIR)
        if not path:
            return
        self.current_file = path
        self._load_file(path)

    def _on_label_select(self, event=None):
        sel = self.lb_labels.curselection()
        self.lb_files.delete(0, tk.END)
        if not sel:
            return
        label = self.lb_labels.get(sel[0])
        folder = os.path.join(RECORDINGS_DIR, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        files.sort()
        for f in files:
            self.lb_files.insert(tk.END, f)
        self._log('Label selected:', label, 'files:', len(files))

    def _on_file_select(self, event=None):
        sel = self.lb_files.curselection()
        if not sel:
            return
        label_sel = self.lb_labels.curselection()
        if not label_sel:
            return
        label = self.lb_labels.get(label_sel[0])
        fname = self.lb_files.get(sel[0])
        path = os.path.join(RECORDINGS_DIR, label, fname)
        self.current_file = path
        self._load_file(path)

    def _load_file(self, path):
        self._log('Selected file:', path)
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self.current_y = y
            self._prepare_processed(y)
            self._update_chart()
        except Exception as e:
            self._log('Load error:', e)

    def _prepare_processed(self, y):
        top_db = float(self.top_db.get())
        self.intervals = librosa.effects.split(y, top_db=top_db)
        self._log('Intervals found:', len(self.intervals))
        if len(self.intervals) == 0:
            self.processed_y = np.array([])
            return
        method = self.method.get()
        if method == 'longest':
            lengths = [(end-start, start, end) for (start,end) in self.intervals]
            lengths.sort(reverse=True)
            _, s, e = lengths[0]
            self.processed_y = y[s:e]
        else:
            parts = [y[s:e] for (s,e) in self.intervals]
            self.processed_y = np.concatenate(parts)
        if len(self.processed_y) / SAMPLE_RATE < float(self.min_duration.get()):
            self._log('Processed shorter than min_duration, discarding')
            self.processed_y = np.array([])
        else:
            self._log('Processed length (s):', len(self.processed_y)/SAMPLE_RATE)

    def _update_chart(self):
        self.ax_orig.cla()
        self.ax_proc.cla()
        if self.current_y is not None:
            t = np.linspace(0, len(self.current_y)/SAMPLE_RATE, len(self.current_y))
            self.ax_orig.plot(t, self.current_y, color='blue')
            # highlight intervals
            for (s,e) in self.intervals:
                self.ax_orig.axvspan(s/SAMPLE_RATE, e/SAMPLE_RATE, color='green', alpha=0.3)
            self.ax_orig.set_title('Original Waveform (with intervals)')
        if self.processed_y is not None and len(self.processed_y) > 0:
            t2 = np.linspace(0, len(self.processed_y)/SAMPLE_RATE, len(self.processed_y))
            self.ax_proc.plot(t2, self.processed_y, color='orange')
            self.ax_proc.set_title('Processed Waveform')
        self.canvas.draw()

    def play_original(self):
        if self.current_y is None:
            self._log('No original loaded')
            return
        sd.play(self.current_y.astype(np.float32), SAMPLE_RATE)
        self._log('Playing original')

    def play_processed(self):
        if self.processed_y is None or len(self.processed_y) == 0:
            self._log('No processed audio to play')
            return
        sd.play(self.processed_y.astype(np.float32), SAMPLE_RATE)
        self._log('Playing processed')

    def save_processed_single(self):
        if not self.current_file or self.processed_y is None or len(self.processed_y) == 0:
            messagebox.showinfo('Info', 'No processed audio to save')
            return
        rel = os.path.relpath(self.current_file, RECORDINGS_DIR)
        if os.sep in rel:
            label, fname = rel.split(os.path.sep, 1)
        else:
            label, fname = 'misc', rel
        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        data_to_write = (self.processed_y * 32767).astype(np.int16)
        sf.write(out_path, data_to_write, SAMPLE_RATE, subtype=SUBTYPE)
        self._log('Saved processed to', out_path)

    def _process_label(self, label):
        folder = os.path.join(RECORDINGS_DIR, label)
        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
        count = 0
        for f in files:
            src = os.path.join(folder, f)
            try:
                y, sr = librosa.load(src, sr=SAMPLE_RATE, mono=True)
                intervals = librosa.effects.split(y, top_db=float(self.top_db.get()))
                if len(intervals) == 0:
                    self._log('No activity:', src)
                    continue
                if self.method.get() == 'longest':
                    lengths = [(end-start, start, end) for (start,end) in intervals]
                    lengths.sort(reverse=True)
                    _, s, e = lengths[0]
                    proc = y[s:e]
                else:
                    parts = [y[s:e] for (s,e) in intervals]
                    proc = np.concatenate(parts)
                if len(proc)/SAMPLE_RATE < float(self.min_duration.get()):
                    self._log('Too short, skip:', src)
                    continue
                out_path = os.path.join(out_dir, f)
                sf.write(out_path, (proc*32767).astype(np.int16), SAMPLE_RATE, subtype=SUBTYPE)
                count += 1
            except Exception as e:
                self._log('Error processing', src, e)
        self._log('Processed label', label, '->', count, 'files')

    def _process_selected_label(self):
        sel = self.lb_labels.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a label first')
            return
        label = self.lb_labels.get(sel[0])
        threading.Thread(target=self._process_label, args=(label,)).start()

    def _process_all_labels(self):
        labels = [d for d in os.listdir(RECORDINGS_DIR) if os.path.isdir(os.path.join(RECORDINGS_DIR, d))]
        threading.Thread(target=self._process_all_worker, args=(labels,)).start()

    def _process_all_worker(self, labels):
        total = 0
        for l in labels:
            self._log('Start label', l)
            self._process_label(l)
            total += 1
        self._log('Done processing', total, 'labels')

if __name__ == '__main__':
    app = DatasetBuilder()
    app.mainloop()