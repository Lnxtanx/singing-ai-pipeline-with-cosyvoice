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

# Spanish translations for all segments
SPANISH_LYRICS = {
    "part1": [
        "Toma mi mano, ven aquí",
        "Un secreto te voy a contar",
        "Con Giramila ven a bailar",
        "En un castillo ven a jugar"
    ],
    "part2": [
        "No habrá reina mala",
        "Ni bruja que espantar",
        "Ser amiga, ser feliz",
        "Ven, vamos a divertirnos"
    ],
    "part3": [
        "Con Giramila juntos caminar",
        "Vamos a percibir",
        "Qué bonito es soñar",
        "Y ver brillar el sol"
    ],
    "part4": [
        "Las flores vamos a regar",
        "Y verlas crecer",
        "Con la amistad que siento",
        "Solo tú y yo",
        "Amigos para siempre vamos a ser"
    ]
}

# Russian translations for all segments
RUSSIAN_LYRICS = {
    "part1": [
        "Возьми мою руку, иди сюда",
        "Секрет я тебе расскажу",
        "С гиромилой будем танцевать",
        "В замке весело играть"
    ],
    "part2": [
        "Не будет злой королевы",
        "Не страшной ведьмы в траве",
        "Быть подругой, быть счастливой",
        "Давай, весело нам жить"
    ],
    "part3": [
        "С гиромилой вместе шагать",
        "Будем мечтать и понимать",
        "Как прекрасно видеть свет",
        "Как солнце ярко светит в лет"
    ],
    "part4": [
        "Цветы мы будем поливать",
        "И за ними наблюдать",
        "С дружбой, что я ощущаю",
        "Только ты и я",
        "друзья навек"
    ]
}

# Italian translations for all segments
ITALIAN_LYRICS = {
    "part1": [
        "Prendi la mia mano, vieni qua",
        "Un segreto ti dirò",
        "Con giramila danzerò",
        "In un castello giocherò"
    ],
    "part2": [
        "Non ci sarà regina cattiva",
        "Né strega da spaventare",
        "Essere amici, essere felici",
        "Vieni, divertiamoci a sognare"
    ],
    "part3": [
        "Con giramila insieme camminare",
        "Scopriremo",
        "Quanto è bello sognare",
        "E vedere il sole brillare"
    ],
    "part4": [
        "Le fiori noi annaffieremo",
        "E crescer li vedremo",
        "Con l'amicizia che sento",
        "Solo io e te",
        "Amici per sempre saremo"
    ]
}

def create_line_chunks_from_json(segment_audio_path, analysis_data, target_lines, output_dir, language="spanish"):
    """
    Create exactly 4 line chunks using JSON word timing data.
    Ensures proper timing, pitch, and tone preservation.
    """
    print("\n" + "="*70)
    print(f"CREATING 4 LINE CHUNKS FROM JSON ANALYSIS ({language.upper()})")
    print("="*70)
    
    audio = AudioSegment.from_wav(segment_audio_path)
    word_timings = analysis_data['transcription']['word_timings_dual_frame']
    
    if len(word_timings) == 0:
        print("❌ No word timings found in analysis")
        return []
    
    # Calculate words per line (divide evenly into 4 lines)
    total_words = len(word_timings)
    words_per_line = total_words // 4
    remainder = total_words % 4
    
    print(f"Total words: {total_words}")
    print(f"Dividing into 4 lines (~{words_per_line} words each)")
    
    # Distribute words across 4 lines
    line_word_ranges = []
    current_idx = 0
    
    for line_num in range(4):
        # Give extra words to first lines if there's a remainder
        line_size = words_per_line + (1 if line_num < remainder else 0)
        end_idx = min(current_idx + line_size - 1, total_words - 1)
        line_word_ranges.append((current_idx, end_idx))
        current_idx = end_idx + 1
    
    # Ensure we have exactly 4 target lines
    if len(target_lines) != 4:
        print(f"⚠️  Warning: Expected 4 {language} lines, got {len(target_lines)}")
        # Pad or truncate
        while len(target_lines) < 4:
            target_lines.append("")
        target_lines = target_lines[:4]
    
    chunks = []
    os.makedirs(output_dir, exist_ok=True)
    
    for line_idx, (start_word_idx, end_word_idx) in enumerate(line_word_ranges):
        if start_word_idx >= len(word_timings):
            break
            
        # Get exact time range from JSON
        start_time_relative = word_timings[start_word_idx]['time_relative']['start']
        end_time_relative = word_timings[end_word_idx]['time_relative']['end']
        
        start_time_ms = start_time_relative * 1000
        end_time_ms = end_time_relative * 1000
        
        # Extract chunk with minimal padding to preserve timing
        padding_ms = 50  # Reduced padding for better sync
        chunk_start = max(0, start_time_ms - padding_ms)
        chunk_end = min(len(audio), end_time_ms + padding_ms)
        
        chunk_audio = audio[chunk_start:chunk_end]
        
        # Save chunk
        chunk_path = os.path.join(output_dir, f"chunk_line_{line_idx + 1}.wav")
        chunk_audio.export(chunk_path, format="wav")
        
        duration = (chunk_end - chunk_start) / 1000.0
        
        # Get Portuguese text and audio features
        portuguese_words = [word_timings[i]['word'] for i in range(start_word_idx, end_word_idx + 1)]
        portuguese_text = ' '.join(portuguese_words)
        
        # Extract average pitch and energy for this line
        avg_pitch = None
        avg_energy = None
        pitch_values = []
        energy_values = []
        
        for i in range(start_word_idx, end_word_idx + 1):
            if 'avg_pitch_hz' in word_timings[i] and word_timings[i]['avg_pitch_hz']:
                pitch_values.append(word_timings[i]['avg_pitch_hz'])
            if 'avg_energy_db' in word_timings[i] and word_timings[i]['avg_energy_db']:
                energy_values.append(word_timings[i]['avg_energy_db'])
        
        if pitch_values:
            avg_pitch = sum(pitch_values) / len(pitch_values)
        if energy_values:
            avg_energy = sum(energy_values) / len(energy_values)
        
        print(f"\n✓ Line {line_idx + 1}: {duration:.2f}s ({start_time_relative:.2f}s - {end_time_relative:.2f}s)")
        print(f"   Words: {start_word_idx}-{end_word_idx} ({end_word_idx - start_word_idx + 1} words)")
        print(f"   Portuguese: {portuguese_text}")
        print(f"   {language.capitalize()}: {target_lines[line_idx]}")
        if avg_pitch:
            print(f"   Avg Pitch: {avg_pitch:.1f} Hz")
        if avg_energy:
            print(f"   Avg Energy: {avg_energy:.1f} dB")
        
        chunks.append({
            'line_number': line_idx + 1,
            'chunk_path': chunk_path,
            'target_text': target_lines[line_idx],
            'duration': duration,
            'time_range': (start_time_relative, end_time_relative),
            'avg_pitch_hz': avg_pitch,
            'avg_energy_db': avg_energy,
            'word_count': end_word_idx - start_word_idx + 1
        })
    
    print(f"\n✓ Created exactly {len(chunks)} line chunks")
    
    return chunks

def generate_target_chunk(cosyvoice, chunk_info, output_path, language="spanish"):
    """
    Generate target language vocals for a single chunk with exact timing and tone preservation.
    Uses reference audio pitch and energy for better matching.
    """
    print(f"\n🎤 Generating {language.capitalize()} for Line {chunk_info['line_number']}...")
    print(f"   Text: {chunk_info['target_text']}")
    print(f"   Duration: {chunk_info['duration']:.2f}s")
    if chunk_info.get('avg_pitch_hz'):
        print(f"   Target Pitch: {chunk_info['avg_pitch_hz']:.1f} Hz")
    
    # Load reference audio
    ref_waveform, sr = torchaudio.load(chunk_info['chunk_path'])
    
    # Convert stereo to mono
    if ref_waveform.shape[0] > 1:
        ref_waveform = torch.mean(ref_waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz (required by CosyVoice)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        prompt_speech_16k = resampler(ref_waveform)
    else:
        prompt_speech_16k = ref_waveform
    
    print(f"   Reference duration: {prompt_speech_16k.shape[1]/16000:.2f}s @ 16kHz")
    
    # Generate target language vocals with same characteristics
    audio_chunks = []
    try:
        # Use speed=1.0 to maintain timing sync
        for model_output in cosyvoice.inference_cross_lingual(
            tts_text=chunk_info['target_text'],
            prompt_speech_16k=prompt_speech_16k,
            stream=False,
            speed=1.0  # Keep original timing
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
        
        # Report timing difference
        ref_duration = chunk_info['duration']
        time_diff = abs(duration - ref_duration)
        
        print(f"✅ Generated: {duration:.2f}s @ {cosyvoice.sample_rate}Hz")
        if time_diff > 0.5:
            print(f"   ⚠️  Time difference: {time_diff:.2f}s from reference")
        else:
            print(f"   ✓ Good sync: {time_diff:.2f}s difference")
        
        return output_path
    else:
        return None

def merge_target_chunks(chunk_outputs, output_path, language="spanish"):
    """
    Merge all target language vocal chunks into final audio file.
    """
    print("\n" + "="*70)
    print(f"MERGING {language.upper()} CHUNKS")
    print("="*70)
    
    merged_audio = AudioSegment.empty()
    
    for i, chunk_path in enumerate(chunk_outputs):
        if chunk_path and os.path.exists(chunk_path):
            chunk = AudioSegment.from_wav(chunk_path)
            merged_audio += chunk
            print(f"✓ Added Line {i + 1}: {len(chunk)/1000:.2f}s")
    
    merged_audio.export(output_path, format="wav")
    total_duration = len(merged_audio) / 1000.0
    
    print(f"\n✅ Merged: {total_duration:.2f}s")
    return output_path

def process_segment(segment_folder, target_lines, part_name, cosyvoice, workspace_root, language="spanish"):
    """
    Process a single segment: create chunks, generate target language, merge.
    """
    print("\n" + "="*70)
    print(f"PROCESSING {part_name.upper()} - {language.upper()}")
    print("="*70)
    
    # Find segment audio file
    segment_files = list(segment_folder.glob("segment_*.wav"))
    if not segment_files:
        print(f"❌ No segment audio found in {segment_folder.name}")
        return None
    
    # Prefer trimmed version if available
    trimmed_files = [f for f in segment_files if 'trimmed' in f.name]
    segment_audio = trimmed_files[0] if trimmed_files else segment_files[0]
    
    # Find analysis JSON
    analysis_files = list(segment_folder.glob("analysis_*.json"))
    if not analysis_files:
        print(f"❌ No analysis JSON found")
        return None
    
    analysis_json = analysis_files[0]
    if trimmed_files:
        trimmed_json = [f for f in analysis_files if 'trimmed' in f.name]
        if trimmed_json:
            analysis_json = trimmed_json[0]
    
    print(f"📁 Audio: {segment_audio.name}")
    print(f"📄 Analysis: {analysis_json.name}")
    
    # Load analysis
    with open(analysis_json, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    duration = analysis['audio_analysis']['duration']
    word_count = len(analysis['transcription']['word_timings_dual_frame'])
    
    print(f"📊 Duration: {duration:.2f}s, Words: {word_count}")
    
    # Create chunks - always create exactly 4 chunks per segment
    chunks_dir = workspace_root / "data" / "temp_chunks" / f"{language}_{part_name}"
    chunks = create_line_chunks_from_json(
        segment_audio_path=str(segment_audio),
        analysis_data=analysis,
        target_lines=target_lines,
        output_dir=str(chunks_dir),
        language=language
    )
    
    if len(chunks) != 4:
        print(f"⚠️  Warning: Expected 4 chunks, got {len(chunks)}")
    
    # Generate target language for each chunk
    output_dir = workspace_root / "data" / "generated_vocals"
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_outputs = []
    for chunk_info in chunks:
        output_path = output_dir / f"{language}_{part_name}_line_{chunk_info['line_number']}.wav"
        result = generate_target_chunk(cosyvoice, chunk_info, str(output_path), language=language)
        chunk_outputs.append(result)
    
    # Merge chunks
    final_output = output_dir / f"{language}_{part_name}_merged.wav"
    merge_target_chunks(chunk_outputs, str(final_output), language=language)
    
    print(f"\n✓ {part_name} complete: {final_output.name}")
    
    return str(final_output)

def main():
    import sys
    
    # Check for language argument
    language = "spanish"  # default
    if len(sys.argv) > 1:
        lang_arg = sys.argv[1].lower()
        if lang_arg in ["spanish", "russian", "italian"]:
            language = lang_arg
        else:
            print(f"❌ Unknown language: {lang_arg}")
            print("Usage: python generate_all_spanish_segments.py [spanish|russian|italian]")
            return
    
    # Select lyrics based on language
    if language == "russian":
        LYRICS = RUSSIAN_LYRICS
    elif language == "italian":
        LYRICS = ITALIAN_LYRICS
    else:
        LYRICS = SPANISH_LYRICS
    
    workspace_root = Path(__file__).parent
    analysis_base = workspace_root / "data" / "analysis"
    
    # Map segment time ranges to parts
    segment_mapping = {
        "6-23s": ("part1", LYRICS["part1"]),
        "22-36s": ("part2", LYRICS["part2"]),
        "35-53s": ("part3", LYRICS["part3"]),
        "52-70s": ("part4", LYRICS["part4"])
    }
    
    # Find all segments
    segment_folders = sorted([f for f in analysis_base.iterdir() 
                             if f.is_dir() and f.name.startswith("segment_analysis_")])
    
    if not segment_folders:
        print("❌ No segment analysis folders found")
        return
    
    print("="*70)
    print(f"MULTI-SEGMENT {language.upper()} VOCAL GENERATION")
    print("="*70)
    print(f"\nFound {len(segment_folders)} segment folders")
    print(f"Target language: {language.capitalize()}")
    
    # Load CosyVoice model once
    print("\n🎶 Loading CosyVoice-300M model...")
    MODEL_NAME = "iic/CosyVoice-300M"
    cosyvoice = CosyVoice(MODEL_NAME)
    print("✅ Model loaded!")
    
    # Process each segment
    results = {}
    
    for folder in segment_folders:
        # Detect which part this is based on segment files
        segment_files = list(folder.glob("segment_*.wav"))
        if not segment_files:
            continue
        
        segment_name = segment_files[0].stem.replace("segment_", "").replace("_trimmed", "")
        
        # Find matching part
        part_name = None
        target_lines = None
        
        for time_range, (pname, tlines) in segment_mapping.items():
            if time_range in segment_name:
                part_name = pname
                target_lines = tlines
                break
        
        if not part_name:
            print(f"\n⚠️  Skipping {folder.name} - no mapping found")
            continue
        
        # Process this segment
        result = process_segment(folder, target_lines, part_name, cosyvoice, workspace_root, language=language)
        if result:
            results[part_name] = result
    
    # Summary
    print("\n" + "="*70)
    print("✓ ALL SEGMENTS PROCESSED!")
    print("="*70)
    print(f"\n📁 Generated {language.capitalize()} vocals:")
    for part, output_file in sorted(results.items()):
        print(f"   {part}: {Path(output_file).name}")
    
    print("\n💡 Note: Part 5 & 6 can reuse Part 1 & 2 (they are identical)")
    print("="*70)

if __name__ == "__main__":
    main()
