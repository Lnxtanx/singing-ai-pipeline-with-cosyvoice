import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from pydub import AudioSegment

# Add CosyVoice to path
repo_root = Path(__file__).resolve().parent / "external" / "CosyVoice"
sys.path.insert(0, str(repo_root))
matcha_path = repo_root / "third_party" / "Matcha-TTS"
sys.path.insert(0, str(matcha_path))

from cosyvoice.cli.cosyvoice import CosyVoice

# --- Config ---
MODEL_NAME = "iic/CosyVoice-300M"
TARGET_LANGUAGE = "spanish"

# Find the latest analysis run
workspace_root = Path(__file__).parent
analysis_base = workspace_root / "data" / "analysis"
run_folders = sorted([f for f in analysis_base.iterdir() if f.is_dir() and f.name.startswith("run_")])

if not run_folders:
    print("❌ Error: No analysis runs found. Please run analyze_vocals_precise.py first!")
    sys.exit(1)

latest_run = run_folders[-1]
print(f"📁 Using analysis run: {latest_run.name}\n")

# Load trimmed analysis
JSON_PATH = latest_run / "analysis_report_trimmed.json"
if not JSON_PATH.exists():
    print(f"❌ Error: Trimmed analysis not found. Please run trim_segment.py first!")
    sys.exit(1)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    analysis = json.load(f)

REFERENCE_AUDIO = Path(analysis["segment_file"])
OUTPUT_DIR = latest_run / f"generated_{TARGET_LANGUAGE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print(f"🎯 GENERATING {TARGET_LANGUAGE.upper()} VOCALS FROM ANALYSIS")
print("="*70)
print(f"Reference audio: {REFERENCE_AUDIO.name}")
print(f"Duration: {analysis['audio_analysis']['duration']:.2f}s")
print(f"Words: {analysis['summary']['total_words']}")
print(f"Language: {analysis['summary']['language']} → {TARGET_LANGUAGE}")
print("="*70)

# --- Load model ---
print("\n🎶 Loading CosyVoice model...")
cosyvoice = CosyVoice(MODEL_NAME)
print("✅ Model ready!\n")

# --- Load reference voice ---
print(f"🎤 Loading reference voice: {REFERENCE_AUDIO}")
ref_waveform, sr = torchaudio.load(str(REFERENCE_AUDIO))

# Convert stereo to mono if needed
if ref_waveform.shape[0] > 1:
    ref_waveform = torch.mean(ref_waveform, dim=0, keepdim=True)

# Resample to 16kHz if needed
if sr != 16000:
    print(f"   Resampling from {sr} Hz to 16000 Hz...")
    resampler = torchaudio.transforms.Resample(sr, 16000)
    ref_waveform = resampler(ref_waveform)

print("✅ Reference voice loaded!\n")

# --- Spanish lyrics aligned with segments ---
# Based on the Portuguese transcription segments
spanish_lyrics = [
    "Toma mi mano, ven aquí",              # Segment 1: 0.00s - 2.20s
    "Un secreto te contaré",               # Segment 2: 3.00s - 6.14s
    "Con giramila a danzar",               # Segment 3: 6.14s - 8.96s
    "En un castillo a jugar",              # Segment 4: 10.00s - 13.22s
    "No habrá reina mala",                 # Segment 5: 13.22s - 16.22s
    "Ni bruja que asuste ya",              # Segment 6: 16.82s - 20.30s
    "Ser amiga, ser feliz"                 # Segment 7: 20.30s - 23.06s
]

segments_info = analysis["transcription"]["segments"]

if len(spanish_lyrics) != len(segments_info):
    print(f"⚠️  Warning: {len(spanish_lyrics)} Spanish lines but {len(segments_info)} segments")

print("="*70)
print("GENERATING VOCAL SEGMENTS")
print("="*70)

generated_segments = []

for i, (spanish_line, segment) in enumerate(zip(spanish_lyrics, segments_info)):
    seg_start = segment['start']
    seg_end = segment['end']
    seg_duration = seg_end - seg_start
    
    print(f"\n🎵 Segment {i+1}/{len(spanish_lyrics)}")
    print(f"   Original: [{seg_start:.2f}s - {seg_end:.2f}s] ({seg_duration:.2f}s)")
    print(f"   Portuguese: {segment['text'].strip()}")
    print(f"   Spanish: {spanish_line}")
    print(f"   Generating...")
    
    # Generate cross-lingual segment
    outputs = []
    try:
        for model_output in cosyvoice.inference_cross_lingual(
            tts_text=spanish_line,
            prompt_speech_16k=ref_waveform,
            stream=False,
            speed=1.0
        ):
            outputs.append(model_output["tts_speech"])
        
        if not outputs:
            print(f"   ⚠️  No output generated for segment {i+1}")
            continue
        
        # Concatenate outputs
        audio = torch.cat(outputs, dim=1)
        
        # Save segment
        seg_path = OUTPUT_DIR / f"segment_{i+1:02d}.wav"
        torchaudio.save(str(seg_path), audio, sample_rate=cosyvoice.sample_rate)
        
        # Calculate generated duration
        generated_duration = audio.shape[1] / cosyvoice.sample_rate
        
        # Time-stretch to match original duration if needed
        duration_ratio = generated_duration / seg_duration
        
        print(f"   ✅ Generated: {seg_path.name}")
        print(f"   Duration: {generated_duration:.2f}s (target: {seg_duration:.2f}s)")
        
        if abs(duration_ratio - 1.0) > 0.15:  # More than 15% difference
            print(f"   ⚠️  Duration mismatch: {duration_ratio:.2f}x")
            print(f"   Applying time-stretch to match original timing...")
            
            # Load as AudioSegment and adjust speed
            audio_seg = AudioSegment.from_wav(str(seg_path))
            
            # Speed adjustment (inverse of duration ratio)
            speed_factor = duration_ratio
            
            # Apply speed change
            adjusted_audio = audio_seg._spawn(
                audio_seg.raw_data,
                overrides={'frame_rate': int(audio_seg.frame_rate * speed_factor)}
            ).set_frame_rate(audio_seg.frame_rate)
            
            # Save adjusted version
            adjusted_path = OUTPUT_DIR / f"segment_{i+1:02d}_adjusted.wav"
            adjusted_audio.export(str(adjusted_path), format="wav")
            
            adjusted_duration = len(adjusted_audio) / 1000.0
            print(f"   ✅ Adjusted: {adjusted_path.name} ({adjusted_duration:.2f}s)")
            
            generated_segments.append({
                'index': i,
                'spanish_text': spanish_line,
                'portuguese_text': segment['text'].strip(),
                'original_start': seg_start,
                'original_end': seg_end,
                'original_duration': seg_duration,
                'file': str(adjusted_path),
                'adjusted': True,
                'final_duration': adjusted_duration
            })
        else:
            generated_segments.append({
                'index': i,
                'spanish_text': spanish_line,
                'portuguese_text': segment['text'].strip(),
                'original_start': seg_start,
                'original_end': seg_end,
                'original_duration': seg_duration,
                'file': str(seg_path),
                'adjusted': False,
                'final_duration': generated_duration
            })
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        continue

print("\n" + "="*70)
print("MERGING SEGMENTS WITH TIMING")
print("="*70)

# Create final audio with proper gaps
final_audio = AudioSegment.silent(duration=0)
current_position = 0.0

for seg_info in generated_segments:
    seg_start = seg_info['original_start']
    
    # Add silence gap if needed
    if seg_start > current_position:
        gap_duration = (seg_start - current_position) * 1000  # Convert to ms
        print(f"\n   Adding {gap_duration/1000:.2f}s silence gap...")
        final_audio += AudioSegment.silent(duration=int(gap_duration))
        current_position = seg_start
    
    # Load and add segment
    segment_audio = AudioSegment.from_wav(seg_info['file'])
    print(f"   Adding segment {seg_info['index']+1}: {seg_info['spanish_text']}")
    final_audio += segment_audio
    
    current_position += len(segment_audio) / 1000.0

# Save final merged audio
final_path = OUTPUT_DIR / f"vocals_{TARGET_LANGUAGE}_complete.wav"
final_audio.export(str(final_path), format="wav")

final_duration = len(final_audio) / 1000.0

print(f"\n✅ Final merged audio saved: {final_path.name}")
print(f"   Duration: {final_duration:.2f}s")
print(f"   Target: {analysis['audio_analysis']['duration']:.2f}s")

# Save generation metadata
metadata = {
    'target_language': TARGET_LANGUAGE,
    'source_analysis': str(JSON_PATH),
    'reference_audio': str(REFERENCE_AUDIO),
    'segments': generated_segments,
    'final_audio': str(final_path),
    'final_duration': final_duration,
    'original_duration': analysis['audio_analysis']['duration']
}

metadata_path = OUTPUT_DIR / "generation_metadata.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Metadata saved: {metadata_path.name}")

print("\n" + "="*70)
print("✨ GENERATION COMPLETE!")
print("="*70)
print(f"\n📂 Output directory: {OUTPUT_DIR}")
print(f"📄 Final audio: {final_path.name}")
print(f"📊 Metadata: {metadata_path.name}")
print(f"\n✓ Spanish vocals generated with proper timing and gaps!")
print("✓ Ready to mix with instrumental track!")
print("="*70)
