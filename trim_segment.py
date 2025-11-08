import os
import json
from pathlib import Path
from pydub import AudioSegment

def trim_segment_to_word(analysis_folder, target_word="feliz"):
    """
    Trim the segment to end at a specific word.
    
    Args:
        analysis_folder: Path to the analysis run folder
        target_word: Word to trim at (default: "feliz")
    """
    analysis_folder = Path(analysis_folder)
    
    # Load the analysis report
    report_path = analysis_folder / "analysis_report.json"
    if not report_path.exists():
        print(f"❌ Error: Analysis report not found at {report_path}")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    # Find the target word
    word_timings = analysis['transcription']['word_timings']
    target_word_data = None
    
    for word in word_timings:
        if word['word'].lower() == target_word.lower():
            target_word_data = word
            break
    
    if not target_word_data:
        print(f"❌ Error: Word '{target_word}' not found in transcription")
        print(f"Available words: {[w['word'] for w in word_timings]}")
        return
    
    # Get the original segment file
    segment_file = Path(analysis['segment_file'])
    if not segment_file.exists():
        print(f"❌ Error: Segment file not found at {segment_file}")
        return
    
    # Calculate trim end time (end of the target word)
    trim_end_time = target_word_data['end']
    
    print(f"\n{'='*70}")
    print(f"TRIMMING SEGMENT TO END AT '{target_word.upper()}'")
    print(f"{'='*70}")
    print(f"\nOriginal segment: {segment_file.name}")
    print(f"Original duration: {analysis['audio_analysis']['duration']:.2f}s")
    print(f"Target word: '{target_word_data['word']}'")
    print(f"Word position: {target_word_data['start']:.3f}s - {target_word_data['end']:.3f}s")
    print(f"Trimming to: {trim_end_time:.3f}s")
    
    # Load audio and trim
    audio = AudioSegment.from_wav(str(segment_file))
    trimmed_audio = audio[:int(trim_end_time * 1000)]  # Convert to milliseconds
    
    # Save trimmed version
    trimmed_filename = segment_file.stem + "_trimmed.wav"
    trimmed_path = analysis_folder / trimmed_filename
    trimmed_audio.export(str(trimmed_path), format="wav")
    
    actual_duration = len(trimmed_audio) / 1000.0
    print(f"\n✓ Trimmed segment saved: {trimmed_filename}")
    print(f"✓ New duration: {actual_duration:.3f}s")
    
    # Filter word segments and timings to only include up to target word
    target_index = target_word_data['index']
    filtered_words = [w for w in word_timings if w['index'] <= target_index]
    filtered_word_segments = [w for w in analysis['word_segments'] if w['index'] <= target_index]
    
    print(f"\n✓ Included words: {len(filtered_words)} (from {word_timings[0]['word']} to {target_word_data['word']})")
    
    # Create updated analysis report
    updated_analysis = analysis.copy()
    updated_analysis['segment_file'] = str(trimmed_path)
    updated_analysis['segment_range']['end'] = analysis['segment_range']['start'] + trim_end_time
    updated_analysis['segment_range']['duration'] = trim_end_time
    updated_analysis['audio_analysis']['duration'] = trim_end_time
    updated_analysis['transcription']['word_timings'] = filtered_words
    updated_analysis['word_segments'] = filtered_word_segments
    updated_analysis['summary']['total_words'] = len(filtered_words)
    
    # Update full text to only include filtered words
    updated_text = ' '.join([w['word'] for w in filtered_words])
    updated_analysis['transcription']['full_text'] = updated_text
    
    # Save updated report
    updated_report_path = analysis_folder / "analysis_report_trimmed.json"
    with open(updated_report_path, 'w', encoding='utf-8') as f:
        json.dump(updated_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated analysis report: analysis_report_trimmed.json")
    
    print(f"\n{'='*70}")
    print(f"TRIMMING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Trimmed segment: {trimmed_filename}")
    print(f"   Duration: {actual_duration:.3f}s (was {analysis['audio_analysis']['duration']:.2f}s)")
    print(f"   Words included: {len(filtered_words)}")
    print(f"   Text: {updated_text}")
    print(f"   Updated report: {updated_report_path.name}")
    print(f"\n✓ Ready for translation with clean ending at '{target_word_data['word']}'!")
    print(f"{'='*70}\n")
    
    return trimmed_path


def main():
    workspace_root = Path(__file__).parent
    
    # Find the most recent analysis folder
    analysis_base = workspace_root / "data" / "analysis"
    
    if not analysis_base.exists():
        print("❌ Error: No analysis folder found")
        print("Please run analyze_vocals_precise.py first!")
        return
    
    # Get all run folders
    run_folders = sorted([f for f in analysis_base.iterdir() if f.is_dir() and f.name.startswith("run_")])
    
    if not run_folders:
        print("❌ Error: No analysis runs found")
        print("Please run analyze_vocals_precise.py first!")
        return
    
    # Use the most recent run
    latest_run = run_folders[-1]
    
    print(f"\n📁 Using analysis run: {latest_run.name}")
    
    # Trim to "feliz"
    trim_segment_to_word(latest_run, target_word="feliz")


if __name__ == "__main__":
    main()
