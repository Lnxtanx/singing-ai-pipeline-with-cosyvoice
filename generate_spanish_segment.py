import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from pydub import AudioSegment

# --- Path Setup ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "external", "CosyVoice"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

matcha_path = os.path.join(repo_root, "third_party", "Matcha-TTS")
if matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice

def create_line_chunks(segment_audio_path, analysis_data, output_dir):
    """
    Split the segment into chunks per line based on word timings.
    Returns list of chunk paths and their corresponding line texts.
    """
    print("\n" + "="*70)
    print("CREATING LINE-BASED CHUNKS")
    print("="*70)
    
    audio = AudioSegment.from_wav(segment_audio_path)
    word_timings = analysis_data['transcription']['word_timings_dual_frame']
    
    # Spanish lyrics split by lines
    spanish_lines = [
        "Toma mi mano, ven aquí",
        "Un secreto te voy a contar",
        "Con Giramila ven a bailar",
        "En un castillo ven a jugar"
    ]
    
    # Portuguese words per line (based on analysis)
    # Line 1: Pegue minha mão vem cá (words 0-4)
    # Line 2: Um segredo vou te contar (words 5-9)
    # Line 3: Com jiramili vem dançar (words 10-13)
    # Line 4: Num castelo vem brincar (words 14-17)
    
    line_word_ranges = [
        (0, 4),   # Line 1: 5 words
        (5, 9),   # Line 2: 5 words
        (10, 13), # Line 3: 4 words
        (14, 17)  # Line 4: 4 words
    ]
    
    chunks = []
    os.makedirs(output_dir, exist_ok=True)
    
    for line_idx, (start_word_idx, end_word_idx) in enumerate(line_word_ranges):
        # Get time range for this line
        start_time_ms = word_timings[start_word_idx]['time_relative']['start'] * 1000
        end_time_ms = word_timings[end_word_idx]['time_relative']['end'] * 1000
        
        # Extract chunk with padding
        padding_ms = 100  # 100ms padding
        chunk_start = max(0, start_time_ms - padding_ms)
        chunk_end = min(len(audio), end_time_ms + padding_ms)
        
        chunk_audio = audio[chunk_start:chunk_end]
        
        # Save chunk
        chunk_path = os.path.join(output_dir, f"chunk_line_{line_idx + 1}.wav")
        chunk_audio.export(chunk_path, format="wav")
        
        duration = (chunk_end - chunk_start) / 1000.0
        print(f"✓ Line {line_idx + 1}: {duration:.2f}s ({start_time_ms/1000:.2f}s - {end_time_ms/1000:.2f}s)")
        print(f"   Portuguese: {' '.join([word_timings[i]['word'] for i in range(start_word_idx, end_word_idx + 1)])}")
        print(f"   Spanish: {spanish_lines[line_idx]}")
        print(f"   Saved: {chunk_path}")
        
        chunks.append({
            'line_number': line_idx + 1,
            'chunk_path': chunk_path,
            'spanish_text': spanish_lines[line_idx],
            'duration': duration,
            'time_range': (start_time_ms / 1000, end_time_ms / 1000)
        })
    
    return chunks


def generate_spanish_chunk(cosyvoice, chunk_info, output_path):
    """
    Generate Spanish vocals for a single chunk.
    """
    print(f"\n🎤 Generating Spanish for Line {chunk_info['line_number']}...")
    print(f"   Reference: {os.path.basename(chunk_info['chunk_path'])}")
    print(f"   Text: {chunk_info['spanish_text']}")
    
    # Load reference audio
    ref_waveform, sr = torchaudio.load(chunk_info['chunk_path'])
    
    # Convert stereo to mono if needed
    if ref_waveform.shape[0] > 1:
        ref_waveform = torch.mean(ref_waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        prompt_speech_16k = resampler(ref_waveform)
    else:
        prompt_speech_16k = ref_waveform
    
    # Generate Spanish vocals
    audio_chunks = []
    try:
        for model_output in cosyvoice.inference_cross_lingual(
            tts_text=chunk_info['spanish_text'],
            prompt_speech_16k=prompt_speech_16k,
            stream=False,
            speed=1.0
        ):
            audio_chunks.append(model_output['tts_speech'])
            print("   .", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None
    
    # Concatenate and save
    if audio_chunks:
        audio = torch.cat(audio_chunks, dim=1)
        duration = audio.shape[1] / cosyvoice.sample_rate
        
        torchaudio.save(output_path, audio, sample_rate=cosyvoice.sample_rate)
        print(f"✅ Generated: {duration:.2f}s @ {cosyvoice.sample_rate}Hz")
        print(f"   Saved: {output_path}")
        
        return output_path
    else:
        print("❌ No audio generated")
        return None


def merge_spanish_chunks(chunk_outputs, output_path, target_sample_rate=22050):
    """
    Merge all Spanish vocal chunks into final audio file.
    """
    print("\n" + "="*70)
    print("MERGING SPANISH VOCAL CHUNKS")
    print("="*70)
    
    merged_audio = AudioSegment.empty()
    
    for i, chunk_path in enumerate(chunk_outputs):
        if chunk_path and os.path.exists(chunk_path):
            chunk = AudioSegment.from_wav(chunk_path)
            merged_audio += chunk
            print(f"✓ Added Line {i + 1}: {len(chunk)/1000:.2f}s")
        else:
            print(f"⚠️  Skipped Line {i + 1}: file not found")
    
    # Export merged audio
    merged_audio.export(output_path, format="wav")
    
    total_duration = len(merged_audio) / 1000.0
    print(f"\n✅ Merged audio saved: {output_path}")
    print(f"   Total duration: {total_duration:.2f}s")
    
    return output_path


def main():
    workspace_root = Path(__file__).parent
    
    # Find the first segment (6-23s trimmed)
    analysis_base = workspace_root / "data" / "analysis"
    segment_folders = sorted([f for f in analysis_base.iterdir() 
                             if f.is_dir() and f.name.startswith("segment_analysis_")])
    
    if not segment_folders:
        print("❌ Error: No segment analysis found")
        print("Please run analyze_specific_segment.py first!")
        return
    
    # Use the first segment (6-23s trimmed)
    target_folder = None
    for folder in segment_folders:
        trimmed_wav = folder / "segment_6-23s_trimmed.wav"
        if trimmed_wav.exists():
            target_folder = folder
            break
    
    if not target_folder:
        print("❌ Error: Trimmed segment not found (segment_6-23s_trimmed.wav)")
        print("Please run trim_remove_words.py first!")
        return
    
    segment_audio = target_folder / "segment_6-23s_trimmed.wav"
    analysis_json = target_folder / "analysis_6-23s_trimmed.json"
    
    print("="*70)
    print("SPANISH VOCAL GENERATION - CHUNK-BASED APPROACH")
    print("="*70)
    print(f"\n📁 Using segment: {target_folder.name}")
    print(f"   Audio: {segment_audio.name}")
    
    # Load analysis
    with open(analysis_json, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    print(f"\n📊 Segment info:")
    print(f"   Duration: {analysis['audio_analysis']['duration']:.2f}s")
    print(f"   Words: {len(analysis['transcription']['word_timings_dual_frame'])}")
    print(f"   Portuguese: {analysis['transcription']['full_text']}")
    
    # Create chunks directory
    chunks_dir = workspace_root / "data" / "temp_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Step 1: Create line-based chunks
    chunks = create_line_chunks(
        segment_audio_path=str(segment_audio),
        analysis_data=analysis,
        output_dir=str(chunks_dir)
    )
    
    print(f"\n✓ Created {len(chunks)} line chunks")
    
    # Step 2: Load CosyVoice model
    print("\n" + "="*70)
    print("LOADING COSYVOICE MODEL")
    print("="*70)
    MODEL_NAME = "iic/CosyVoice-300M"
    cosyvoice = CosyVoice(MODEL_NAME)
    print("✅ Model loaded successfully!")
    
    # Step 3: Generate Spanish for each chunk
    print("\n" + "="*70)
    print("GENERATING SPANISH VOCALS PER LINE")
    print("="*70)
    
    output_dir = workspace_root / "data" / "generated_vocals"
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_outputs = []
    for chunk_info in chunks:
        output_path = output_dir / f"spanish_line_{chunk_info['line_number']}.wav"
        result = generate_spanish_chunk(cosyvoice, chunk_info, str(output_path))
        chunk_outputs.append(result)
    
    # Step 4: Merge all chunks
    final_output = output_dir / "spanish_segment_1_merged.wav"
    merge_spanish_chunks(chunk_outputs, str(final_output))
    
    print("\n" + "="*70)
    print("✓ GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"   Individual lines: {output_dir}")
    for i, path in enumerate(chunk_outputs):
        if path:
            print(f"     - Line {i+1}: {os.path.basename(path)}")
    print(f"   Merged file: {final_output.name}")
    print("="*70)


if __name__ == "__main__":
    main()
