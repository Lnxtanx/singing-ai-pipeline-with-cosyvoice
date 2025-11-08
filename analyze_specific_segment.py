import os
import json
from pathlib import Path
from openai import OpenAI
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from datetime import datetime

def extract_and_analyze_specific_segment(vocals_path, start_time, end_time, output_dir, target_lyrics):
    """
    Extract a specific segment and analyze with dual time frames:
    1. Relative time (0-based within segment)
    2. Absolute time (position in original file)
    """
    
    print("="*70)
    print("EXTRACTING SPECIFIC VOCAL SEGMENT")
    print("="*70)
    print(f"Source: {vocals_path}")
    print(f"Time range: {start_time:.2f}s - {end_time:.2f}s")
    print(f"Duration: {end_time - start_time:.2f}s")
    print(f"Target lyrics: {target_lyrics}")
    
    # Extract segment
    audio = AudioSegment.from_wav(vocals_path)
    segment = audio[start_time * 1000:end_time * 1000]
    
    segment_path = output_dir / f"segment_{start_time:.0f}-{end_time:.0f}s.wav"
    segment.export(str(segment_path), format="wav")
    
    print(f"✓ Extracted segment: {segment_path.name}")
    
    # Analyze the segment
    print("\n" + "="*70)
    print("ANALYZING SEGMENT AUDIO")
    print("="*70)
    
    y, sr = librosa.load(str(segment_path), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Tempo and beat
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item()) if tempo.size == 1 else float(tempo[0])
    else:
        tempo = float(tempo)
    
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    print(f"Duration: {duration:.2f}s")
    print(f"Tempo: {tempo:.1f} BPM")
    print(f"Beats: {len(beat_times)}")
    
    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times_relative = librosa.frames_to_time(onset_frames, sr=sr)
    
    print(f"Onsets detected: {len(onset_times_relative)}")
    
    # Energy analysis
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_contour = []
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length)
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_contour.append({"time_relative": float(times[t]), "pitch_hz": float(pitch)})
    
    # Transcribe with Whisper
    print("\n" + "="*70)
    print("TRANSCRIBING WITH WORD TIMESTAMPS")
    print("="*70)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(str(segment_path), "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    
    print(f"Transcribed: {transcription.text}")
    
    # Build dual time frame word data
    word_timings_dual = []
    
    if hasattr(transcription, 'words') and transcription.words:
        print(f"\n{'#':<4} {'Word':<15} {'Relative':<20} {'Absolute':<20}")
        print("-"*70)
        
        for i, word in enumerate(transcription.words):
            time_relative_start = float(word.start)
            time_relative_end = float(word.end)
            
            # Calculate absolute time (in original file)
            time_absolute_start = start_time + time_relative_start
            time_absolute_end = start_time + time_relative_end
            
            word_data = {
                "index": i,
                "word": word.word.strip(),
                "time_relative": {
                    "start": time_relative_start,
                    "end": time_relative_end,
                    "duration": time_relative_end - time_relative_start
                },
                "time_absolute": {
                    "start": time_absolute_start,
                    "end": time_absolute_end,
                    "duration": time_absolute_end - time_absolute_start
                }
            }
            
            # Add pitch and energy
            word_pitches = [p['pitch_hz'] for p in pitch_contour 
                          if time_relative_start <= p['time_relative'] <= time_relative_end]
            if word_pitches:
                word_data['avg_pitch_hz'] = float(np.mean(word_pitches))
            
            energy_mask = (rms_times >= time_relative_start) & (rms_times <= time_relative_end)
            if np.any(energy_mask):
                word_data['avg_energy_db'] = float(np.mean(rms_db[energy_mask]))
            
            word_timings_dual.append(word_data)
            
            print(f"{i+1:<4} {word.word:<15} "
                  f"{time_relative_start:.2f}-{time_relative_end:.2f}s    "
                  f"{time_absolute_start:.2f}-{time_absolute_end:.2f}s")
    
    # Create analysis report
    analysis_data = {
        "segment_info": {
            "source_file": str(vocals_path),
            "extracted_segment": str(segment_path),
            "time_range_absolute": {
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time
            },
            "target_lyrics": target_lyrics
        },
        "audio_analysis": {
            "duration": float(duration),
            "sample_rate": int(sr),
            "tempo": tempo,
            "beat_times_relative": beat_times.tolist(),
            "onset_times_relative": onset_times_relative.tolist(),
            "pitch_contour": pitch_contour[:100]  # Limit for JSON size
        },
        "transcription": {
            "full_text": transcription.text,
            "language": transcription.language if hasattr(transcription, 'language') else "unknown",
            "word_timings_dual_frame": word_timings_dual
        },
        "note": "Dual time frames: 'relative' is 0-based within segment, 'absolute' is position in original file for merging"
    }
    
    # Save analysis
    analysis_path = output_dir / f"analysis_{start_time:.0f}-{end_time:.0f}s.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis saved: {analysis_path.name}")
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6, color='blue')
    plt.title(f"Vocal Segment: {start_time:.0f}s-{end_time:.0f}s (Absolute Time)", fontweight='bold')
    for word in word_timings_dual:
        plt.axvline(word['time_relative']['start'], color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.ylabel("Amplitude")
    
    # Subplot 2: Spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram with Word Boundaries", fontweight='bold')
    plt.ylim(0, 3000)
    
    # Subplot 3: Energy
    plt.subplot(3, 1, 3)
    plt.plot(rms_times, rms_db, color='purple', linewidth=1.5)
    plt.title("Energy (RMS) Over Time", fontweight='bold')
    plt.xlabel("Time (s) - Relative")
    plt.ylabel("Energy (dB)")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    viz_path = output_dir / f"visualization_{start_time:.0f}-{end_time:.0f}s.png"
    plt.savefig(str(viz_path), dpi=150)
    plt.close()
    
    print(f"✓ Visualization saved: {viz_path.name}")
    
    return analysis_data


def main():
    workspace_root = Path(__file__).parent
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it using: $env:OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Input
    vocals_file = workspace_root / "data" / "separated" / "vocals.wav"
    
    if not vocals_file.exists():
        print(f"ERROR: Vocals file not found at {vocals_file}")
        return
    
    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = workspace_root / "data" / "analysis" / f"segment_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Target segment parameters - PART 6
    START_TIME = 93.0   # Starts at 1:33 (93 seconds)
    END_TIME = 107.0    # Ends at 1:47 (107 seconds)
    TARGET_LYRICS = "Part 6: 4 lines (1:33 to 1:47) - Não terá rainha má / Nem bruxa pra espantar / Ser amiga, ser feliz / Vem, vamos nos divertir"
    
    print("\n" + "="*70)
    print("SPECIFIC SEGMENT ANALYSIS WITH DUAL TIME FRAMES")
    print("="*70)
    print(f"Output folder: {output_dir.name}\n")
    
    # Extract and analyze
    analysis_data = extract_and_analyze_specific_segment(
        vocals_path=str(vocals_file),
        start_time=START_TIME,
        end_time=END_TIME,
        output_dir=output_dir,
        target_lyrics=TARGET_LYRICS
    )
    
    # Summary
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Segment: {START_TIME:.0f}s - {END_TIME:.0f}s (duration: {END_TIME - START_TIME:.0f}s)")
    print(f"   Words transcribed: {len(analysis_data['transcription']['word_timings_dual_frame'])}")
    print(f"   Tempo: {analysis_data['audio_analysis']['tempo']:.1f} BPM")
    print(f"\n📁 Output:")
    print(f"   Folder: {output_dir}")
    print(f"   Segment audio: segment_{START_TIME:.0f}-{END_TIME:.0f}s.wav")
    print(f"   Analysis data: analysis_{START_TIME:.0f}-{END_TIME:.0f}s.json")
    print(f"   Visualization: visualization_{START_TIME:.0f}-{END_TIME:.0f}s.png")
    print(f"\n✓ Dual time frames recorded:")
    print(f"   - Relative (0-based): For TTS generation")
    print(f"   - Absolute (original position): For merging with background music")
    print("="*70)


if __name__ == "__main__":
    main()
