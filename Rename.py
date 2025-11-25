import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

def select_folder():
    folder_path.set(filedialog.askdirectory())
    if folder_path.get():
        lbl_folder.config(text=folder_path.get())

def rename_files():
    src_folder = folder_path.get()
    if not src_folder:
        messagebox.showwarning("Warning", "Please select a folder first!")
        return

    # Tạo folder mới
    base_name = os.path.basename(src_folder)
    dst_folder = os.path.join(os.path.dirname(src_folder), f"{base_name}_new")
    
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Duyệt các subfolder
    for subdir, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(subdir, src_folder)
        new_subdir = os.path.join(dst_folder, rel_path)
        os.makedirs(new_subdir, exist_ok=True)

        for file in files:
            if file.endswith(".wav") and "_" in file:
                parts = file.rsplit("_", 1)
                if len(parts) == 2:
                    name, num_ext = parts
                    num, ext = os.path.splitext(num_ext)
                    new_name = f"{name}.{num}{ext}"
                    shutil.copy2(os.path.join(subdir, file), os.path.join(new_subdir, new_name))

    messagebox.showinfo("Success", f"Renaming done! Files saved in:\n{dst_folder}")

# GUI
root = tk.Tk()
root.title("Batch Rename WAV Files")

folder_path = tk.StringVar()

btn_select = tk.Button(root, text="Select Folder", command=select_folder, width=20)
btn_select.pack(pady=10)

lbl_folder = tk.Label(root, text="No folder selected", fg="blue")
lbl_folder.pack(pady=5)

btn_rename = tk.Button(root, text="Rename Files", command=rename_files, width=20, bg="lightgreen")
btn_rename.pack(pady=20)

root.geometry("400x200")
root.mainloop()
