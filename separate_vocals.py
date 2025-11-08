# src/separate_vocals.py
import os
import subprocess
import sys

def separate_audio(input_path, output_dir="data/separated"):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        sys.executable, "-m", "demucs.separate",
        "-n", "mdx_extra",
        input_path,
        "--out", output_dir
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    input_file = r"data/raw/Amigos para Sempre - Giramille ｜ Desenho Animado Musical.wav"
    separate_audio(input_file)
