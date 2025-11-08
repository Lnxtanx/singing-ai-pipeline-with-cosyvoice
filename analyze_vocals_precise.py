import os
import json
from pathlib import Path
from openai import OpenAI
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from datetime import datetime

def auto_detect_vocal_region(input_audio, min_duration=15, max_duration=25):
    """
    Automatically detect the first continuous vocal region in the audio.
    
    Args:
        input_audio: Path to input audio file
        min_duration: Minimum duration for segment (seconds)
        max_duration: Maximum duration for segment (seconds)
    
    Returns:
        (start_time, end_time) tuple
    """
    print("\n🔍 Auto-detecting vocal region...")
    
    y, sr = librosa.load(input_audio, sr=16000)
    
    # Compute energy
    energy = librosa.feature.rms(y=y, hop_length=512)[0]
    
    # Find frames with significant energy (above 70th percentile)
    threshold = np.percentile(energy, 70)
    active_frames = np.where(energy > threshold)[0]
    
    if len(active_frames) == 0:
        print("⚠️  No vocal activity detected, using default range")
        return 5, 25
    
    # Convert frames to time
    frame_times = librosa.frames_to_time(active_frames, sr=sr, hop_length=512)
    
    # Find first continuous region
    start_time = frame_times[0]
    
    # Look for continuous activity up to max_duration
    end_time = min(start_time + max_duration, frame_times[-1])
    
    # Ensure minimum duration
    if end_time - start_time < min_duration:
        end_time = min(start_time + min_duration, librosa.get_duration(y=y, sr=sr))
    
    print(f"✓ Detected vocal region: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")
    
    return float(start_time), float(end_time)


def extract_segment(input_audio, start_sec, end_sec, output_path):
    """
    Extract a segment from the audio file.
    
    Args:
        input_audio: Path to input audio file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        output_path: Where to save the extracted segment
    """
    audio = AudioSegment.from_wav(input_audio)
    segment = audio[start_sec * 1000:end_sec * 1000]
    segment.export(output_path, format="wav")
    print(f"✓ Extracted segment: {start_sec:.2f}s to {end_sec:.2f}s -> {output_path}")
    return output_path


def hz_to_note(freq):
    """Convert frequency in Hz to musical note name."""
    if freq <= 0 or freq > 8000:
        return None
    try:
        note = librosa.hz_to_note(freq)
        return note
    except:
        return None


def analyze_audio_detailed(audio_path):
    """
    Perform detailed analysis of audio: pitch, tempo, rhythm, energy, etc.
    """
    print("\n" + "="*70)
    print("DETAILED AUDIO ANALYSIS")
    print("="*70)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"\n📊 Basic Info:")
    print(f"   Duration: {duration:.3f} seconds")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Total Samples: {len(y)}")
    
    # Tempo and beat analysis
    print(f"\n🎵 Rhythm Analysis:")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Ensure tempo is a scalar value
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item()) if tempo.size == 1 else float(tempo[0])
    else:
        tempo = float(tempo)
    
    print(f"   Tempo: {tempo:.2f} BPM")
    print(f"   Number of beats: {len(beat_times)}")
    print(f"   Beat times: {beat_times[:10].tolist()}" + (" ..." if len(beat_times) > 10 else ""))
    
    # Pitch/frequency analysis
    print(f"\n🎼 Pitch Analysis:")
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get pitch contour over time
    pitch_contour = []
    hop_length = 512
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length)
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_contour.append({"time": float(times[t]), "pitch_hz": float(pitch)})
    
    # Convert pitches to musical notes
    notes = []
    note_histogram = {}
    
    if pitch_contour:
        avg_pitch = np.mean([p['pitch_hz'] for p in pitch_contour])
        min_pitch = np.min([p['pitch_hz'] for p in pitch_contour])
        max_pitch = np.max([p['pitch_hz'] for p in pitch_contour])
        print(f"   Average Pitch: {avg_pitch:.2f} Hz")
        print(f"   Pitch Range: {min_pitch:.2f} Hz - {max_pitch:.2f} Hz")
        print(f"   Pitch Contour Points: {len(pitch_contour)}")
        
        # Extract musical notes
        for p in pitch_contour:
            note = hz_to_note(p['pitch_hz'])
            if note:
                notes.append(note)
                note_histogram[note] = note_histogram.get(note, 0) + 1
        
        if note_histogram:
            dominant_notes = sorted(note_histogram.items(), key=lambda x: -x[1])[:5]
            print(f"   Dominant Notes: {[f'{n}({c})' for n, c in dominant_notes]}")
    
    # Onset detection (word/syllable boundaries)
    print(f"\n🔊 Onset Detection (Word/Syllable Boundaries):")
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, 
        wait=1, pre_avg=1, post_avg=1, 
        pre_max=1, post_max=1
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = librosa.onset.onset_strength(y=y, sr=sr)
    
    print(f"   Detected onsets: {len(onset_times)}")
    print(f"   Onset times: {onset_times.tolist()}")
    
    # Energy/volume analysis over time
    print(f"\n📈 Energy/Volume Analysis:")
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Normalize RMS
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    energy_contour = [{"time": float(t), "energy_db": float(e)} 
                      for t, e in zip(rms_times, rms_db)]
    
    print(f"   Energy contour points: {len(energy_contour)}")
    print(f"   Average energy: {np.mean(rms_db):.2f} dB")
    
    # Detect silence/gaps
    print(f"\n🔇 Silence Detection:")
    threshold = np.percentile(rms, 15)  # Bottom 15% is silence
    silence_mask = rms < threshold
    
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_gap:
            gap_start = rms_times[i]
            in_gap = True
        elif not is_silent and in_gap:
            gap_end = rms_times[i]
            if gap_end - gap_start > 0.15:  # Gaps longer than 0.15s
                gaps.append({
                    "start": float(gap_start),
                    "end": float(gap_end),
                    "duration": float(gap_end - gap_start)
                })
            in_gap = False
    
    print(f"   Silent gaps detected: {len(gaps)}")
    for i, gap in enumerate(gaps[:5]):
        print(f"   Gap {i+1}: {gap['start']:.3f}s - {gap['end']:.3f}s (duration: {gap['duration']:.3f}s)")
    if len(gaps) > 5:
        print(f"   ... and {len(gaps) - 5} more gaps")
    
    # Spectral features
    print(f"\n🌈 Spectral Analysis:")
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    print(f"   Average Spectral Centroid: {np.mean(spectral_centroids):.2f} Hz")
    print(f"   Average Spectral Rolloff: {np.mean(spectral_rolloff):.2f} Hz")
    
    # Prepare dominant notes for return
    dominant_notes_list = []
    if note_histogram:
        dominant_notes_list = [{"note": n, "count": c} for n, c in sorted(note_histogram.items(), key=lambda x: -x[1])[:10]]
    
    return {
        "duration": float(duration),
        "sample_rate": int(sr),
        "tempo": float(tempo),
        "beat_times": beat_times.tolist(),
        "onset_times": onset_times.tolist(),
        "num_syllables_estimate": len(onset_times),
        "pitch_contour": pitch_contour,
        "energy_contour": energy_contour,
        "gaps": gaps,
        "spectral_centroid_mean": float(np.mean(spectral_centroids)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "dominant_notes": dominant_notes_list,
        "raw_audio_data": {
            "y_shape": len(y),
            "sample_rate": int(sr)
        }
    }


def transcribe_with_word_timestamps(audio_path, client, onset_times=None):
    """
    Transcribe audio using Whisper with precise word-level timestamps.
    Cross-validates with onset detection if provided.
    """
    print("\n" + "="*70)
    print("TRANSCRIPTION WITH WORD TIMESTAMPS")
    print("="*70)
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    
    full_text = transcription.text
    print(f"\n📝 Full Transcription:")
    print(f"   {full_text}")
    
    # Extract word-level timing with onset alignment
    word_timings = []
    if hasattr(transcription, 'words') and transcription.words:
        print(f"\n⏱️  Word-Level Timestamps:")
        print(f"   {'#':<4} {'Word':<20} {'Start':<10} {'End':<10} {'Duration':<10} {'Offset':<10}")
        print(f"   {'-'*70}")
        
        for i, word in enumerate(transcription.words):
            duration = word.end - word.start
            
            word_data = {
                "index": i,
                "word": word.word.strip(),
                "start": float(word.start),
                "end": float(word.end),
                "duration": float(duration)
            }
            
            # Cross-validate with onset detection
            if onset_times is not None and len(onset_times) > 0:
                nearest_onset = min(onset_times, key=lambda o: abs(o - word.start))
                onset_offset = word.start - nearest_onset
                word_data["onset_offset"] = float(onset_offset)
                
                print(f"   {i+1:<4} {word.word:<20} {word.start:<10.3f} {word.end:<10.3f} {duration:<10.3f} {onset_offset:+.3f}")
                
                if abs(onset_offset) > 0.3:
                    print(f"        ⚠️  Large offset detected - may need realignment")
            else:
                print(f"   {i+1:<4} {word.word:<20} {word.start:<10.3f} {word.end:<10.3f} {duration:<10.3f} {'N/A':<10}")
            
            word_timings.append(word_data)
    
    # Extract segment-level info
    segments = []
    if hasattr(transcription, 'segments') and transcription.segments:
        print(f"\n📑 Segment-Level Info:")
        for seg in transcription.segments:
            segments.append({
                "text": seg.text.strip(),
                "start": float(seg.start),
                "end": float(seg.end)
            })
            print(f"   [{seg.start:.2f}s - {seg.end:.2f}s]: {seg.text}")
    
    return {
        "full_text": full_text,
        "word_timings": word_timings,
        "segments": segments,
        "language": transcription.language if hasattr(transcription, 'language') else "unknown"
    }


def enrich_word_timings_with_features(word_timings, audio_path, pitch_contour, energy_contour, rms_times, rms_db):
    """
    Add per-word pitch and energy features for synthesis control.
    """
    print("\n" + "="*70)
    print("ENRICHING WORD TIMINGS WITH AUDIO FEATURES")
    print("="*70)
    
    for word in word_timings:
        start = word['start']
        end = word['end']
        
        # Calculate average energy for this word
        energy_mask = (rms_times >= start) & (rms_times <= end)
        if np.any(energy_mask):
            word['avg_energy_db'] = float(np.mean(rms_db[energy_mask]))
            word['max_energy_db'] = float(np.max(rms_db[energy_mask]))
        else:
            word['avg_energy_db'] = None
            word['max_energy_db'] = None
        
        # Calculate average pitch for this word
        word_pitches = [p['pitch_hz'] for p in pitch_contour if start <= p['time'] <= end]
        if word_pitches:
            word['avg_pitch_hz'] = float(np.mean(word_pitches))
            word['min_pitch_hz'] = float(np.min(word_pitches))
            word['max_pitch_hz'] = float(np.max(word_pitches))
            word['pitch_range_hz'] = word['max_pitch_hz'] - word['min_pitch_hz']
            
            # Convert to notes
            avg_note = hz_to_note(word['avg_pitch_hz'])
            word['avg_note'] = avg_note
        else:
            word['avg_pitch_hz'] = None
            word['min_pitch_hz'] = None
            word['max_pitch_hz'] = None
            word['pitch_range_hz'] = None
            word['avg_note'] = None
        
        # Format output safely
        pitch_str = f"{word['avg_pitch_hz']:.1f}" if word['avg_pitch_hz'] else "N/A"
        note_str = word['avg_note'] if word['avg_note'] else "N/A"
        energy_str = f"{word['avg_energy_db']:.1f}" if word['avg_energy_db'] else "N/A"
        
        print(f"   '{word['word']}': pitch={pitch_str} Hz, note={note_str}, energy={energy_str} dB")
    
    return word_timings


def create_word_segments(audio_path, word_timings, output_dir):
    """
    Split audio into individual word segments for precise translation.
    """
    print("\n" + "="*70)
    print("CREATING WORD SEGMENTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio = AudioSegment.from_wav(audio_path)
    word_files = []
    
    for i, word_info in enumerate(word_timings):
        word = word_info['word']
        start_ms = int(word_info['start'] * 1000)
        end_ms = int(word_info['end'] * 1000)
        
        # Add small padding (50ms before and after)
        padding = 50
        start_ms = max(0, start_ms - padding)
        end_ms = min(len(audio), end_ms + padding)
        
        word_segment = audio[start_ms:end_ms]
        
        # Clean filename
        safe_word = word.replace('/', '_').replace('\\', '_').replace(' ', '_')
        filename = f"word_{i:03d}_{safe_word}.wav"
        filepath = os.path.join(output_dir, filename)
        
        word_segment.export(filepath, format="wav")
        
        word_file_info = {
            "index": i,
            "word": word,
            "file": filename,
            "filepath": filepath,
            "start": word_info['start'],
            "end": word_info['end'],
            "duration": word_info['duration']
        }
        
        # Copy over enriched features
        if 'avg_pitch_hz' in word_info:
            word_file_info['avg_pitch_hz'] = word_info['avg_pitch_hz']
            word_file_info['avg_note'] = word_info['avg_note']
        if 'avg_energy_db' in word_info:
            word_file_info['avg_energy_db'] = word_info['avg_energy_db']
        
        word_files.append(word_file_info)
        
        print(f"   ✓ Word {i+1}: '{word}' -> {filename}")
    
    print(f"\n✓ Created {len(word_files)} word segments in: {output_dir}")
    
    return word_files


def create_visual_diagnostics(audio_path, onset_times, beat_times, word_timings, output_dir):
    """
    Generate visual diagnostic plots for analysis.
    """
    print("\n" + "="*70)
    print("GENERATING VISUAL DIAGNOSTICS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    y, sr = librosa.load(audio_path, sr=None)
    
    # Plot 1: Waveform with onsets
    plt.figure(figsize=(14, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.6, color='blue')
    plt.title("Waveform with Onset Detection", fontsize=14, fontweight='bold')
    
    for onset in onset_times:
        plt.axvline(onset, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    waveform_path = os.path.join(output_dir, "01_waveform_onsets.png")
    plt.savefig(waveform_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {waveform_path}")
    
    # Plot 2: Spectrogram with word boundaries
    plt.figure(figsize=(14, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram with Word Boundaries", fontsize=14, fontweight='bold')
    
    for word in word_timings:
        plt.axvline(word['start'], color='lime', linestyle='-', alpha=0.7, linewidth=1.5)
        plt.axvline(word['end'], color='orange', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 4000)
    plt.tight_layout()
    spectrogram_path = os.path.join(output_dir, "02_spectrogram_words.png")
    plt.savefig(spectrogram_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {spectrogram_path}")
    
    # Plot 3: Energy contour
    plt.figure(figsize=(14, 5))
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    plt.plot(times, rms, color='purple', linewidth=1.5)
    plt.title("Energy (RMS) Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    energy_path = os.path.join(output_dir, "03_energy_contour.png")
    plt.savefig(energy_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {energy_path}")
    
    print(f"\n✓ Visual diagnostics saved to: {output_dir}")
    
    return {
        "waveform_onsets": waveform_path,
        "spectrogram_words": spectrogram_path,
        "energy_contour": energy_path
    }


def create_markdown_report(output_path, analysis_data, visual_paths):
    """
    Create a human-readable Markdown report.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 🎵 Vocal Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 Summary\n\n")
        summary = analysis_data['summary']
        f.write(f"- **Duration:** {analysis_data['segment_range']['duration']:.2f}s\n")
        f.write(f"- **Tempo:** {summary['tempo_bpm']:.1f} BPM\n")
        f.write(f"- **Language:** {summary['language']}\n")
        f.write(f"- **Total Words:** {summary['total_words']}\n")
        f.write(f"- **Estimated Syllables:** {summary['total_syllables_estimate']}\n")
        f.write(f"- **Silent Gaps:** {summary['silent_gaps']}\n\n")
        
        # Dominant notes
        if analysis_data['audio_analysis'].get('dominant_notes'):
            f.write("### 🎼 Dominant Musical Notes\n\n")
            for note_info in analysis_data['audio_analysis']['dominant_notes'][:5]:
                f.write(f"- **{note_info['note']}**: {note_info['count']} occurrences\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 📝 Transcription\n\n")
        f.write(f"```\n{analysis_data['transcription']['full_text']}\n```\n\n")
        
        f.write("---\n\n")
        f.write("## ⏱️ Word-Level Timing\n\n")
        f.write("| # | Word | Start (s) | End (s) | Duration (s) | Pitch (Hz) | Note | Energy (dB) |\n")
        f.write("|---|------|-----------|---------|--------------|------------|------|-------------|\n")
        
        for word in analysis_data['transcription']['word_timings']:
            pitch = f"{word['avg_pitch_hz']:.1f}" if word.get('avg_pitch_hz') else "N/A"
            note = word.get('avg_note', 'N/A')
            energy = f"{word['avg_energy_db']:.1f}" if word.get('avg_energy_db') else "N/A"
            
            f.write(f"| {word['index']+1} | {word['word']} | {word['start']:.3f} | "
                   f"{word['end']:.3f} | {word['duration']:.3f} | {pitch} | {note} | {energy} |\n")
        
        f.write("\n---\n\n")
        f.write("## 📈 Visual Diagnostics\n\n")
        
        if visual_paths:
            for title, path in visual_paths.items():
                rel_path = Path(path).name
                f.write(f"### {title.replace('_', ' ').title()}\n\n")
                f.write(f"![{title}]({rel_path})\n\n")
        
        f.write("---\n\n")
        f.write("## 🔇 Detected Silence Gaps\n\n")
        
        gaps = analysis_data['audio_analysis']['gaps']
        if gaps:
            for i, gap in enumerate(gaps):
                f.write(f"{i+1}. **{gap['start']:.3f}s - {gap['end']:.3f}s** (duration: {gap['duration']:.3f}s)\n")
        else:
            f.write("*No significant silence gaps detected.*\n")
        
        f.write("\n---\n\n")
        f.write("*End of Report*\n")
    
    print(f"✓ Markdown report saved: {output_path}")


def save_analysis_report(output_path, analysis_data):
    """
    Save comprehensive analysis report as JSON.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis report saved: {output_path}")


def main():
    workspace_root = Path(__file__).parent
    
    # Create timestamped run folder for data lineage
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = workspace_root / "data" / "analysis" / f"run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"📁 Analysis Run Folder: {run_folder}")
    print("="*70)
    
    # Input: full vocals file
    vocals_file = workspace_root / "data" / "separated" / "vocals.wav"
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it using: $env:OPENAI_API_KEY='your-api-key-here'")
        return
    
    if not vocals_file.exists():
        print(f"ERROR: Vocals file not found at {vocals_file}")
        return
    
    # Step 1: Auto-detect vocal region
    print("\n" + "="*70)
    print("STEP 1: AUTO-DETECTING VOCAL REGION")
    print("="*70)
    
    start_time, end_time = auto_detect_vocal_region(str(vocals_file), min_duration=15, max_duration=25)
    
    segment_output = run_folder / f"segment_{start_time:.1f}-{end_time:.1f}s.wav"
    extract_segment(str(vocals_file), start_time, end_time, str(segment_output))
    
    # Step 2: Detailed audio analysis
    print("\n" + "="*70)
    print("STEP 2: ANALYZING AUDIO CHARACTERISTICS")
    print("="*70)
    
    audio_analysis = analyze_audio_detailed(str(segment_output))
    
    # Step 3: Transcribe with word timestamps
    print("\n" + "="*70)
    print("STEP 3: TRANSCRIBING WITH WORD TIMESTAMPS")
    print("="*70)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    transcription_data = transcribe_with_word_timestamps(
        str(segment_output), 
        client,
        onset_times=audio_analysis['onset_times']
    )
    
    # Step 4: Enrich word timings with audio features
    print("\n" + "="*70)
    print("STEP 4: ENRICHING WORD DATA WITH PITCH & ENERGY")
    print("="*70)
    
    # Load RMS data for enrichment
    y, sr = librosa.load(str(segment_output), sr=None)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    enriched_word_timings = enrich_word_timings_with_features(
        transcription_data['word_timings'],
        str(segment_output),
        audio_analysis['pitch_contour'],
        audio_analysis['energy_contour'],
        rms_times,
        rms_db
    )
    
    transcription_data['word_timings'] = enriched_word_timings
    
    # Step 5: Create individual word segments
    print("\n" + "="*70)
    print("STEP 5: SPLITTING INTO WORD SEGMENTS")
    print("="*70)
    
    word_segments_dir = run_folder / "word_segments"
    word_files = create_word_segments(
        str(segment_output),
        enriched_word_timings,
        str(word_segments_dir)
    )
    
    # Step 6: Generate visual diagnostics
    print("\n" + "="*70)
    print("STEP 6: CREATING VISUAL DIAGNOSTICS")
    print("="*70)
    
    visual_dir = run_folder / "visualizations"
    visual_paths = create_visual_diagnostics(
        str(segment_output),
        audio_analysis['onset_times'],
        audio_analysis['beat_times'],
        enriched_word_timings,
        str(visual_dir)
    )
    
    # Step 7: Compile complete analysis report
    print("\n" + "="*70)
    print("STEP 7: GENERATING ANALYSIS REPORTS")
    print("="*70)
    
    complete_analysis = {
        "run_id": timestamp,
        "source_file": str(vocals_file),
        "segment_file": str(segment_output),
        "segment_range": {
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time
        },
        "audio_analysis": audio_analysis,
        "transcription": transcription_data,
        "word_segments": word_files,
        "visual_diagnostics": visual_paths,
        "summary": {
            "total_words": len(enriched_word_timings),
            "total_syllables_estimate": audio_analysis['num_syllables_estimate'],
            "tempo_bpm": audio_analysis['tempo'],
            "silent_gaps": len(audio_analysis['gaps']),
            "language": transcription_data['language'],
            "dominant_notes": audio_analysis.get('dominant_notes', [])[:5]
        }
    }
    
    # Save JSON report
    report_json_path = run_folder / "analysis_report.json"
    save_analysis_report(str(report_json_path), complete_analysis)
    
    # Save Markdown report
    report_md_path = run_folder / "analysis_report.md"
    create_markdown_report(str(report_md_path), complete_analysis, visual_paths)
    
    # Final summary
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Run ID: {timestamp}")
    print(f"   Original file: {vocals_file.name}")
    print(f"   Analyzed segment: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")
    print(f"   Language detected: {transcription_data['language']}")
    print(f"   Total words: {len(enriched_word_timings)}")
    print(f"   Tempo: {audio_analysis['tempo']:.1f} BPM")
    
    if audio_analysis.get('dominant_notes'):
        top_notes = [n['note'] for n in audio_analysis['dominant_notes'][:3]]
        print(f"   Top notes: {', '.join(top_notes)}")
    
    print(f"   Silent gaps: {len(audio_analysis['gaps'])}")
    print(f"   Word segments created: {len(word_files)}")
    print(f"\n📁 Output folder:")
    print(f"   {run_folder}")
    print(f"\n📄 Reports:")
    print(f"   JSON: {report_json_path.name}")
    print(f"   Markdown: {report_md_path.name}")
    print(f"\n📊 Visualizations:")
    print(f"   {visual_dir}")
    print(f"\n✓ Ready for translation! Each word is isolated with precise timing, pitch, and energy data.")
    print("="*70)


if __name__ == "__main__":
    main()
