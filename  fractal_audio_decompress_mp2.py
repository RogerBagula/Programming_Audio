#!/usr/bin/env python3
"""
Fractal audio decompressor (.bl -> WAV) with:
- Accepts .bl fractal files
- Progress output (iterations + block counter)
- Crash logging
- Automatic save directory ~/FractalAudio/
- Multiprocessing for speed
- Matches compressor's new format
"""

import os
import json
import numpy as np
import soundfile as sf
import multiprocessing as mp
import traceback
from datetime import datetime


# ---------- Logging ----------

LOG_DIR = os.path.expanduser("~/FractalAudio")
LOG_FILE = os.path.join(LOG_DIR, "decompress_log.txt")

os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------- Block maker ----------

def make_blocks(data, block_size):
    N = data.shape[1]
    blocks = []
    for start in range(0, N - block_size, block_size):
        blocks.append((start, data[:, start:start + block_size]))
    return blocks


# ---------- Worker for multiprocessing ----------

def apply_entry(args):
    entry, domains, block_size = args
    try:
        range_idx = entry["range_index"]
        dom_idx = entry["domain_index"]
        A = np.array(entry["A"], dtype=np.float32)
        b = np.array(entry["b"], dtype=np.float32)

        start = range_idx * block_size
        end = start + block_size

        if dom_idx >= len(domains):
            return None

        D_big = domains[dom_idx]
        D = D_big[:, ::2]  # downsample

        R_hat = A @ D + b[:, None]
        return (start, end, R_hat, range_idx)

    except Exception as e:
        log(f"Error in entry {entry['range_index']}: {e}")
        log(traceback.format_exc())
        return None


# ---------- Fractal decompression ----------

def fractal_decompress_mp(bl_path, iterations=20):
    log(f"Loading fractal file: {bl_path}")

    with open(bl_path, "r") as f:
        bl = json.load(f)

    block_size = bl["block_size"]
    samplerate = bl["samplerate"]
    entries = bl["entries"]

    # Determine total length from highest range index
    max_range = max(e["range_index"] for e in entries)
    data_length = (max_range + 1) * block_size

    log(f"Reconstructing {data_length} samples, block size {block_size}")
    log(f"Total blocks: {len(entries)}")

    data = np.zeros((4, data_length), dtype=np.float32)

    for it in range(iterations):
        log(f"Iteration {it + 1} / {iterations}")

        domains = [blk for _, blk in make_blocks(data, block_size * 2)]

        args_list = [(entry, domains, block_size) for entry in entries]

        with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
            for idx, res in enumerate(pool.imap_unordered(apply_entry, args_list)):
                if res is None:
                    continue

                start, end, R_hat, range_idx = res
                if end <= data.shape[1]:
                    data[:, start:end] = R_hat

                # Block counter
                if idx % 50 == 0:
                    log(f"Iteration {it + 1}: block {idx + 1} / {len(entries)}")

    return data, samplerate


# ---------- Save WAV ----------

def save_wav(path, data, samplerate):
    data = data.T  # (N, 4)
    sf.write(path, data, samplerate)
    log(f"Saved WAV file: {path}")


# ---------- Main ----------

def main():
    import tkinter as tk
    from tkinter import filedialog

    log("Fractal audio decompressor started.")

    root = tk.Tk()
    root.withdraw()
    bl_file = filedialog.askopenfilename(
        title="Select .bl fractal file",
        filetypes=[("Fractal audio", "*.bl")]
    )

    if not bl_file:
        log("No .bl file selected. Exiting.")
        return

    try:
        data, sr = fractal_decompress_mp(bl_file, iterations=20)
    except Exception as e:
        log(f"Decompression error: {e}")
        log(traceback.format_exc())
        return

    base_name = os.path.splitext(os.path.basename(bl_file))[0]
    output_wav = os.path.join(LOG_DIR, base_name + "_reconstructed.wav")

    save_wav(output_wav, data, sr)

    log("Done.")


if __name__ == "__main__":
    main()
