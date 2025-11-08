import os
import sys
import torch
import torchaudio
import numpy as np

# --- Path Setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "external", "CosyVoice"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

matcha_path = os.path.join(repo_root, "third_party", "Matcha-TTS")
if matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice

# --- Directories ---
CHUNKS_DIR = "data/separated/chunks"       # Directory where split_vocals.py saved chunks
TRANSCRIPTS_DIR = "data/transcripts"
OUTPUT_DIR = "data/generated_vocals"
MODEL_NAME = "iic/CosyVoice-300M"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Language Map ---
languages = {
    "spanish": "lyrics_spanish.txt",
    "italian": "lyrics_italian.txt",
    "russian": "lyrics_russian.txt",
}

# --- Load Model ---
print("🎶 Loading CosyVoice model... please wait.")
cosyvoice = CosyVoice(MODEL_NAME)
print("✅ Model loaded successfully!")

# --- Process Each Chunk ---
chunk_files = sorted(
    [os.path.join(CHUNKS_DIR, f) for f in os.listdir(CHUNKS_DIR) if f.endswith(".wav")]
)

if not chunk_files:
    print("❌ No chunks found in:", CHUNKS_DIR)
    sys.exit(1)

print(f"🎤 Found {len(chunk_files)} vocal chunks to process.\n")

# --- Generate Vocals for Each Language ---
for lang, filename in languages.items():
    lyric_path = os.path.join(TRANSCRIPTS_DIR, filename)
    if not os.path.exists(lyric_path):
        print(f"⚠️ Skipping {lang} — {lyric_path} not found.")
        continue

    with open(lyric_path, "r", encoding="utf-8") as f:
        lyrics = f.read().strip()

    print(f"\n🎧 Generating vocals in {lang.capitalize()} using {len(chunk_files)} chunks...")

    for idx, chunk_path in enumerate(chunk_files):
        print(f"🎵 Processing chunk {idx + 1}/{len(chunk_files)}: {os.path.basename(chunk_path)}")

        # Load chunk
        ref_waveform, sr = torchaudio.load(chunk_path)

        # --- Fix stereo to mono (CosyVoice expects 1 channel) ---
        if ref_waveform.shape[0] > 1:
            ref_waveform = torch.mean(ref_waveform, dim=0, keepdim=True)

        # --- Resample to 16kHz (required by CosyVoice) ---
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            prompt_speech_16k = resampler(ref_waveform)
        else:
            prompt_speech_16k = ref_waveform

        # --- Generate vocals for this chunk ---
        audio_chunks = []
        try:
            for model_output in cosyvoice.inference_cross_lingual(
                tts_text=lyrics,
                prompt_speech_16k=prompt_speech_16k,
                stream=False,
                speed=1.0
            ):
                audio_chunks.append(model_output['tts_speech'])
        except Exception as e:
            print(f"❌ Error processing chunk {idx}: {e}")
            continue

        # --- Concatenate and save chunk output ---
        if audio_chunks:
            audio = torch.cat(audio_chunks, dim=1)
            output_path = os.path.join(OUTPUT_DIR, f"vocals_{lang}_chunk_{idx}.wav")
            torchaudio.save(output_path, audio, sample_rate=cosyvoice.sample_rate)
            print(f"✅ Saved: {output_path}")
        else:
            print(f"⚠️ No output generated for chunk {idx}")

print("\n🎉 All vocal chunks generated successfully!")
print(f"🗂️  Output directory: {os.path.abspath(OUTPUT_DIR)}")
