#!/usr/bin/env python3
import os
import struct
import numpy as np
from scipy.io import wavfile
import tkinter as tk
from tkinter import filedialog

# ============================================================
# FAST FRACTAL COMPRESSION (Google-style vectorized)
# ============================================================

def fast_compress_audio(signal, range_size=32, domain_size=64, step_size=64):
    num_samples = len(signal)
    num_ranges = num_samples // range_size

    domain_indices = np.arange(0, num_samples - domain_size + 1, step_size)
    num_domains = len(domain_indices)

    domains_raw = np.array([signal[i:i + domain_size] for i in domain_indices])
    domains_down = domains_raw.reshape(num_domains, range_size, domain_size // range_size).mean(axis=2)

    d_means = domains_down.mean(axis=1, keepdims=True)
    d_vars = domains_down.var(axis=1, keepdims=True)
    d_vars[d_vars == 0] = 1e-9
    d_dev = domains_down - d_means

    ranges = signal[:num_ranges * range_size].reshape(num_ranges, range_size)
    r_means = ranges.mean(axis=1, keepdims=True)
    r_dev = ranges - r_means

    print(f"Encoding {num_ranges} blocks using {num_domains} domains...")

    fractal_codes = []

    for r_idx in range(num_ranges):
        r_block = ranges[r_idx]
        r_d = r_dev[r_idx]

        cov = np.dot(d_dev, r_d) / range_size
        s_array = cov / d_vars.squeeze()
        s_array = np.clip(s_array, -0.95, 0.95)

        o_array = r_means[r_idx] - s_array * d_means.squeeze()

        estimates = s_array[:, None] * domains_down + o_array[:, None]
        errors = np.sum((r_block - estimates) ** 2, axis=1)

        best_idx = np.argmin(errors)

        fractal_codes.append((
            int(domain_indices[best_idx]),
            float(s_array[best_idx]),
            float(o_array[best_idx])
        ))

    return fractal_codes

# ============================================================
# SAVE / LOAD FRACTAL FILE (.frac)
# ============================================================

def save_fractal_file(filepath, fractal_codes, total_samples, sample_rate, range_size, domain_size):
    header_fmt = "<IIHHI"
    num_blocks = len(fractal_codes)

    with open(filepath, "wb") as f:
        f.write(struct.pack(header_fmt, total_samples, sample_rate, range_size, domain_size, num_blocks))

        block_fmt = "<Iff"
        for domain_idx, s, o in fractal_codes:
            f.write(struct.pack(block_fmt, domain_idx, s, o))

def load_fractal_file(filepath):
    header_fmt = "<IIHHI"
    header_size = struct.calcsize(header_fmt)

    with open(filepath, "rb") as f:
        header_data = f.read(header_size)
        total_samples, sample_rate, range_size, domain_size, num_blocks = struct.unpack(header_fmt, header_data)

        block_fmt = "<Iff"
        block_size = struct.calcsize(block_fmt)

        fractal_codes = []
        for _ in range(num_blocks):
            domain_idx, s, o = struct.unpack(block_fmt, f.read(block_size))
            fractal_codes.append((domain_idx, s, o))

    return fractal_codes, total_samples, sample_rate, range_size, domain_size

# ============================================================
# FAST DECOMPRESSION
# ============================================================

def fast_decompress_audio(fractal_codes, total_samples, range_size, domain_size, iterations=8):
    reconstructed_signal = np.zeros(total_samples, dtype=np.float32)
    num_ranges = total_samples // range_size

    for _ in range(iterations):
        prev = reconstructed_signal.copy()

        for r_idx in range(num_ranges):
            domain_idx, s, o = fractal_codes[r_idx]

            d_block = prev[domain_idx:domain_idx + domain_size]
            d_down = d_block.reshape(range_size, domain_size // range_size).mean(axis=1)

            reconstructed_signal[r_idx*range_size:(r_idx+1)*range_size] = s * d_down + o

    return reconstructed_signal

# ============================================================
# REAL FILE PICKER MENU
# ============================================================

def pick_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def pick_save_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    return filedialog.asksaveasfilename(title=title, filetypes=filetypes)

def main():
    print("FAST FRACTAL AUDIO CODEC")
    print("1 = Compress WAV → FRAC")
    print("2 = Decompress FRAC → WAV")
    mode = input("Choose option: ").strip()

    if mode == "1":
        in_path = pick_file("Select WAV file", [("WAV files", "*.wav")])
        if not in_path:
            print("No file selected.")
            return

        out_path = pick_save_file("Save FRAC file as", [("Fractal files", "*.frac")])
        if not out_path:
            print("No save location selected.")
            return

        sr, audio = wavfile.read(in_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio))

        R_SIZE = 32
        D_SIZE = 64
        STEP = 64

        pad_len = (D_SIZE - (len(audio) % D_SIZE)) % D_SIZE
        audio = np.pad(audio, (0, pad_len), 'constant')

        print("Compressing...")
        codes = fast_compress_audio(audio, R_SIZE, D_SIZE, STEP)

        save_fractal_file(out_path, codes, len(audio), sr, R_SIZE, D_SIZE)
        print(f"Saved compressed file: {out_path}")

    elif mode == "2":
        in_path = pick_file("Select FRAC file", [("Fractal files", "*.frac")])
        if not in_path:
            print("No file selected.")
            return

        out_path = pick_save_file("Save WAV file as", [("WAV files", "*.wav")])
        if not out_path:
            print("No save location selected.")
            return

        codes, total_samples, sr, R_SIZE, D_SIZE = load_fractal_file(in_path)

        print("Decompressing...")
        recon = fast_decompress_audio(codes, total_samples, R_SIZE, D_SIZE, iterations=8)

        recon_scaled = (recon * 32767).astype(np.int16)
        wavfile.write(out_path, sr, recon_scaled)

        print(f"Saved reconstructed WAV: {out_path}")

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
