import os
import json
from pathlib import Path
from pydub import AudioSegment
import numpy as np

def load_segment_analysis(segment_folder):
    """
    Load the analysis JSON to get absolute time frame information.
    """
    analysis_files = list(segment_folder.glob("analysis_*.json"))
    if not analysis_files:
        return None
    
    # Prefer trimmed version
    trimmed_json = [f for f in analysis_files if 'trimmed' in f.name]
    analysis_path = trimmed_json[0] if trimmed_json else analysis_files[0]
    
    with open(analysis_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_vocals_to_timeline(workspace_root, language="spanish"):
    """
    Merge all vocal segment parts into a single timeline based on absolute timestamps.
    """
    print("="*70)
    print(f"MERGING {language.upper()} SEGMENTS TO ORIGINAL TIMELINE")
    print("="*70)
    
    analysis_base = workspace_root / "data" / "analysis"
    generated_vocals_dir = workspace_root / "data" / "generated_vocals"
    
    # Map segments to their time ranges and vocal files
    segment_info = []
    
    # Find all segment folders and their info
    segment_folders = sorted([f for f in analysis_base.iterdir() 
                             if f.is_dir() and f.name.startswith("segment_analysis_")])
    
    part_mapping = {
        "6-23s": "part1",
        "22-36s": "part2",
        "35-53s": "part3",
        "52-70s": "part4",
        "79-93s": "part5",  # Reuse part1
        "93-107s": "part6"  # Reuse part2
    }
    
    for folder in segment_folders:
        analysis = load_segment_analysis(folder)
        if not analysis:
            continue
        
        # Get absolute time range
        time_range = analysis['segment_info']['time_range_absolute']
        start_time = time_range['start']
        end_time = time_range['end']
        
        # Determine which part this is
        segment_files = list(folder.glob("segment_*.wav"))
        if not segment_files:
            continue
        
        segment_name = segment_files[0].stem.replace("segment_", "").replace("_trimmed", "")
        
        part_name = None
        for time_key, pname in part_mapping.items():
            if time_key in segment_name:
                part_name = pname
                break
        
        if not part_name:
            print(f"⚠️  Skipping {folder.name} - no part mapping found")
            continue
        
        # Check if this is a reused part (part5 = part1, part6 = part2)
        if part_name == "part5":
            vocal_file = generated_vocals_dir / f"{language}_part1_merged.wav"
            actual_part = "part1 (reused as part5)"
        elif part_name == "part6":
            vocal_file = generated_vocals_dir / f"{language}_part2_merged.wav"
            actual_part = "part2 (reused as part6)"
        else:
            vocal_file = generated_vocals_dir / f"{language}_{part_name}_merged.wav"
            actual_part = part_name
        
        if not vocal_file.exists():
            print(f"⚠️  {language.capitalize()} file not found: {vocal_file.name}")
            continue
        
        segment_info.append({
            'part_name': part_name,
            'actual_part': actual_part,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'vocal_file': str(vocal_file)
        })
    
    # Sort by start time
    segment_info.sort(key=lambda x: x['start_time'])
    
    print(f"\n📊 Found {len(segment_info)} segments to merge:\n")
    for seg in segment_info:
        print(f"   {seg['actual_part']}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s ({seg['duration']:.1f}s)")
    
    if not segment_info:
        print("\n❌ No segments to merge!")
        return None
    
    # Calculate total duration needed
    max_end_time = max(seg['end_time'] for seg in segment_info)
    total_duration_ms = int(max_end_time * 1000)
    
    print(f"\n🎵 Creating timeline: 0s - {max_end_time:.1f}s ({max_end_time:.1f}s total)")
    
    # Create silent base track
    print("\n📝 Building merged track...")
    base_track = AudioSegment.silent(duration=total_duration_ms)
    
    # Overlay each vocal segment at its absolute position
    for seg in segment_info:
        print(f"\n   Adding {seg['actual_part']}:")
        print(f"   Position: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        
        # Load vocal audio
        vocal_audio = AudioSegment.from_wav(seg['vocal_file'])
        vocal_duration = len(vocal_audio) / 1000.0
        
        print(f"   Vocal duration: {vocal_duration:.2f}s")
        print(f"   Target duration: {seg['duration']:.2f}s")
        
        # Check duration match
        time_diff = abs(vocal_duration - seg['duration'])
        if time_diff > 1.0:
            print(f"   ⚠️  Large duration mismatch: {time_diff:.2f}s difference")
            # Optionally stretch/compress to match
            if vocal_duration > seg['duration']:
                # Speed up slightly
                speed_factor = vocal_duration / seg['duration']
                vocal_audio = vocal_audio.speedup(playback_speed=speed_factor)
                print(f"   Adjusted speed by {speed_factor:.2f}x to fit timeline")
        
        # Calculate overlay position
        overlay_position_ms = int(seg['start_time'] * 1000)
        
        # Overlay at the absolute position
        base_track = base_track.overlay(vocal_audio, position=overlay_position_ms)
        
        print(f"   ✓ Overlaid at {seg['start_time']:.2f}s")
    
    # Export merged track
    output_path = workspace_root / "data" / "generated_vocals" / f"{language}_full_vocals_merged.wav"
    base_track.export(str(output_path), format="wav")
    
    final_duration = len(base_track) / 1000.0
    
    print("\n" + "="*70)
    print("✓ MERGE COMPLETE!")
    print("="*70)
    print(f"\n📁 Output: {output_path.name}")
    print(f"🎵 Duration: {final_duration:.2f}s")
    print(f"📊 Segments merged: {len(segment_info)}")
    print("\n💡 This track uses absolute timestamps from the original vocals")
    print("   Ready to mix with instrumental track!")
    print("="*70)
    
    return str(output_path)

def create_timing_report(workspace_root):
    """
    Create a detailed timing report for verification.
    """
    analysis_base = workspace_root / "data" / "analysis"
    output_path = workspace_root / "data" / "generated_vocals" / "TIMING_REPORT.txt"
    
    segment_folders = sorted([f for f in analysis_base.iterdir() 
                             if f.is_dir() and f.name.startswith("segment_analysis_")])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("SPANISH VOCALS TIMELINE REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("SEGMENT TIMELINE (Absolute Timestamps):\n")
        f.write("-"*70 + "\n\n")
        
        for folder in segment_folders:
            analysis = load_segment_analysis(folder)
            if not analysis:
                continue
            
            time_range = analysis['segment_info']['time_range_absolute']
            f.write(f"Segment: {folder.name}\n")
            f.write(f"  Start: {time_range['start']:.2f}s\n")
            f.write(f"  End: {time_range['end']:.2f}s\n")
            f.write(f"  Duration: {time_range['duration']:.2f}s\n")
            
            if 'target_lyrics' in analysis['segment_info']:
                f.write(f"  Lyrics: {analysis['segment_info']['target_lyrics']}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("REUSED SEGMENTS:\n")
        f.write("-"*70 + "\n")
        f.write("Part 5 (79-93s): Reuses Part 1 Spanish vocals\n")
        f.write("Part 6 (93-107s): Reuses Part 2 Spanish vocals\n")
        f.write("\n")
    
    print(f"\n📄 Timing report saved: {output_path.name}")

def main():
    """
    Merge vocal segments into a complete timeline
    Usage: python merge_spanish_timeline.py [language]
    Example: python merge_spanish_timeline.py russian
    """
    import sys
    
    workspace_root = Path(__file__).parent
    
    # Get language from command line argument (default: spanish)
    language = sys.argv[1] if len(sys.argv) > 1 else "spanish"
    
    print("\n" + "="*70)
    print(f"{language.upper()} VOCALS MERGER")
    print("="*70)
    
    # Create timing report
    create_timing_report(workspace_root)
    
    # Merge all segments
    output_file = merge_vocals_to_timeline(workspace_root, language)
    
    if output_file:
        print(f"\n✅ Success! {language.capitalize()} vocals ready at:")
        print(f"   {output_file}")
    else:
        print("\n❌ Merge failed!")

if __name__ == "__main__":
    main()
