from pydub import AudioSegment
from pathlib import Path

def slow_down_audio(input_path, output_path, speed_factor=0.8):
    """
    Slow down audio by adjusting playback speed.
    speed_factor < 1.0 means slower (0.8 = 20% slower)
    """
    print("="*70)
    print("SLOWING DOWN SPANISH VOCALS")
    print("="*70)
    
    print(f"\nInput: {input_path}")
    print(f"Speed factor: {speed_factor}x (20% slower)")
    
    # Load audio
    audio = AudioSegment.from_wav(input_path)
    original_duration = len(audio) / 1000.0
    
    print(f"Original duration: {original_duration:.2f}s")
    
    # Slow down by changing frame rate
    # Lower frame rate = slower playback
    slowed_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    })
    
    # Convert back to original sample rate to maintain quality
    slowed_audio = slowed_audio.set_frame_rate(audio.frame_rate)
    
    new_duration = len(slowed_audio) / 1000.0
    
    print(f"New duration: {new_duration:.2f}s")
    print(f"Difference: +{new_duration - original_duration:.2f}s")
    
    # Export
    slowed_audio.export(output_path, format="wav")
    
    print(f"\n✅ Saved: {output_path}")
    print("="*70)
    
    return output_path

def main():
    workspace_root = Path(__file__).parent
    
    input_file = workspace_root / "data" / "generated_vocals" / "spanish_full_vocals_merged.wav"
    output_file = workspace_root / "data" / "generated_vocals" / "spanish_full_vocals_merged_slowed.wav"
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("Please run merge_spanish_timeline.py first!")
        return
    
    # Slow down by 20% (speed_factor = 0.8)
    slow_down_audio(str(input_file), str(output_file), speed_factor=0.8)
    
    print(f"\n🎵 Slowed Spanish vocals ready!")
    print(f"   Original: {input_file.name}")
    print(f"   Slowed: {output_file.name}")

if __name__ == "__main__":
    main()
