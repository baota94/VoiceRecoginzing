#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import soundfile as sf
import os

SAMPLE_RATE = 16000
RECORD_FILE = "temp_record.wav"

class MicTestGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Microphone Test")
        self.geometry("500x200")

        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self):
        # Buttons
        frame = ttk.Frame(self)
        frame.pack(pady=20)

        ttk.Button(frame, text="Record", command=self._record).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Play", command=self._play).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Mic Info", command=self._mic_info).pack(side=tk.LEFT, padx=5)

        # Status label
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, padx=10, pady=20)
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

    def _mic_info(self):
        try:
            default_input = sd.default.device[0]
            info = sd.query_devices(default_input, 'input')
            samplerate = info.get('default_samplerate', 0)
            name = info.get('name', 'Unknown')
            # test supported dtypes
            dtypes = ['int16', 'int32', 'float32']
            supported = []
            for dt in dtypes:
                try:
                    with sd.RawInputStream(samplerate=int(samplerate),
                                           channels=1, dtype=dt, device=default_input):
                        supported.append(dt)
                except:
                    pass
            msg = f"Mic: {name}, SampleRate: {samplerate}, Supported: {supported}"
        except Exception as e:
            msg = f"Error detecting mic: {e}"
        self._update_status(msg)

    def _update_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

if __name__ == "__main__":
    app = MicTestGUI()
    app.mainloop()
