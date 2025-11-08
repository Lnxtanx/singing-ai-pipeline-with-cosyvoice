from pydub import AudioSegment
from pathlib import Path

def mix_vocals_with_instrumental(vocals_path, instrumental_path, output_path, vocals_volume=0, instrumental_volume=-2):
    """
    Mix Spanish vocals with instrumental track.
    
    Args:
        vocals_path: Path to Spanish vocals
        instrumental_path: Path to instrumental track
        output_path: Where to save the mixed output
        vocals_volume: Volume adjustment in dB (0 = no change)
        instrumental_volume: Volume adjustment in dB (-2 = slightly quieter)
    """
    print("\n" + "="*70)
    print(f"MIXING: {Path(vocals_path).name}")
    print("="*70)
    
    # Load audio files
    print("\n📁 Loading files...")
    vocals = AudioSegment.from_wav(vocals_path)
    instrumental = AudioSegment.from_wav(instrumental_path)
    
    vocals_duration = len(vocals) / 1000.0
    instrumental_duration = len(instrumental) / 1000.0
    
    print(f"   Vocals: {vocals_duration:.2f}s")
    print(f"   Instrumental: {instrumental_duration:.2f}s")
    
    # Adjust volumes
    if vocals_volume != 0:
        vocals = vocals + vocals_volume
        print(f"   Vocals volume: {vocals_volume:+.1f} dB")
    
    if instrumental_volume != 0:
        instrumental = instrumental + instrumental_volume
        print(f"   Instrumental volume: {instrumental_volume:+.1f} dB")
    
    # Ensure both tracks have the same length (use longer duration)
    max_duration = max(len(vocals), len(instrumental))
    
    if len(vocals) < max_duration:
        # Pad vocals with silence
        silence_duration = max_duration - len(vocals)
        vocals = vocals + AudioSegment.silent(duration=silence_duration)
        print(f"   Padded vocals with {silence_duration/1000:.2f}s silence")
    
    if len(instrumental) < max_duration:
        # Pad instrumental with silence
        silence_duration = max_duration - len(instrumental)
        instrumental = instrumental + AudioSegment.silent(duration=silence_duration)
        print(f"   Padded instrumental with {silence_duration/1000:.2f}s silence")
    
    # Mix the tracks
    print("\n🎵 Mixing tracks...")
    mixed = vocals.overlay(instrumental)
    
    mixed_duration = len(mixed) / 1000.0
    print(f"   Mixed duration: {mixed_duration:.2f}s")
    
    # Export
    print(f"\n💾 Exporting to: {output_path}")
    mixed.export(output_path, format="wav")
    
    print("✅ Mix complete!")
    
    return output_path

def main():
    """
    Mix vocals with instrumental track
    Usage: python mix_spanish_with_instrumental.py [language]
    Example: python mix_spanish_with_instrumental.py russian
    """
    import sys
    
    workspace_root = Path(__file__).parent
    
    # Get language from command line argument (default: spanish)
    language = sys.argv[1] if len(sys.argv) > 1 else "spanish"
    
    # Input files
    vocals_normal = workspace_root / "data" / "generated_vocals" / f"{language}_full_vocals_merged.wav"
    vocals_slowed = workspace_root / "data" / "generated_vocals" / f"{language}_full_vocals_merged_slowed.wav"
    instrumental = workspace_root / "data" / "separated" / "instrumental.wav"
    
    # Output files
    output_normal = workspace_root / "data" / "generated_vocals" / f"{language}_final_mix_normal.wav"
    output_slowed = workspace_root / "data" / "generated_vocals" / f"{language}_final_mix_slowed.wav"
    
    # Check if files exist
    if not instrumental.exists():
        print(f"❌ Error: Instrumental file not found at {instrumental}")
        print("\nLooking for instrumental in separated folder...")
        
        # Try to find instrumental
        separated_dir = workspace_root / "data" / "separated"
        if separated_dir.exists():
            inst_files = list(separated_dir.glob("*instrumental*.wav")) + list(separated_dir.glob("*accompaniment*.wav"))
            if inst_files:
                instrumental = inst_files[0]
                print(f"✓ Found: {instrumental.name}")
            else:
                print("❌ No instrumental file found in separated folder")
                return
        else:
            print("❌ Separated folder not found")
            return
    
    print("="*70)
    print(f"MIXING {language.upper()} VOCALS WITH INSTRUMENTAL")
    print("="*70)
    
    # Mix normal version
    if vocals_normal.exists():
        print("\n🎼 Processing NORMAL speed version...")
        mix_vocals_with_instrumental(
            vocals_path=str(vocals_normal),
            instrumental_path=str(instrumental),
            output_path=str(output_normal),
            vocals_volume=0,      # Keep vocals at original volume
            instrumental_volume=-2  # Slightly quieter instrumental
        )
    else:
        print(f"\n⚠️  Skipping normal version - file not found: {vocals_normal.name}")
    
    # Mix slowed version
    if vocals_slowed.exists():
        print("\n🎼 Processing SLOWED speed version...")
        mix_vocals_with_instrumental(
            vocals_path=str(vocals_slowed),
            instrumental_path=str(instrumental),
            output_path=str(output_slowed),
            vocals_volume=0,      # Keep vocals at original volume
            instrumental_volume=-2  # Slightly quieter instrumental
        )
    else:
        print(f"\n⚠️  Skipping slowed version - file not found: {vocals_slowed.name}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ ALL MIXES COMPLETE!")
    print("="*70)
    
    if vocals_normal.exists():
        print(f"\n📁 Normal speed mix: {output_normal.name}")
    
    if vocals_slowed.exists():
        print(f"📁 Slowed speed mix: {output_slowed.name}")
    
    print(f"\n🎵 Final {language} song{'s' if vocals_slowed.exists() else ''} ready!")
    print("="*70)

if __name__ == "__main__":
    main()
