#!/usr/bin/env python3
"""
Example usage of the auto-tune function with presets
"""

from pitch_correction_utils import main

def example_usage():
    """Demonstrate different ways to use the auto-tune function"""
    
    input_file = "input_audio.m4a"  # Your audio file
    
    print("=== Auto-tune Examples ===\n")
    
    # Example 1: Using a preset (easiest way)
    print("1. Using preset 'natural' for subtle correction:")
    try:
        result = main(
            vocals_file=input_file,
            autotune_preset="natural",
            plot=True
        )
        print("   ✓ Created: input_audio_pitch_corrected.wav")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Using preset 'classic' for that classic autotune sound:")
    try:
        result = main(
            vocals_file=input_file,
            autotune_preset="classic",
            plot=True
        )
        print("   ✓ Created: input_audio_pitch_corrected.wav")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Using preset 'c_major' for scale-based correction:")
    try:
        result = main(
            vocals_file=input_file,
            autotune_preset="c_major",
            plot=True
        )
        print("   ✓ Created: input_audio_pitch_corrected.wav")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Using individual parameters (more control):")
    try:
        result = main(
            vocals_file=input_file,
            correction_strength=0.5,
            adaptive_strength=True,
            smooth_transitions=True,
            plot=True
        )
        print("   ✓ Created: input_audio_pitch_corrected.wav")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n=== Available Presets ===")
    presets = [
        "natural", "gentle", "balanced", "strong", "classic", 
        "ultra_subtle", "c_major", "a_minor"
    ]
    for preset in presets:
        print(f"• {preset}")
    
    print("\n=== Usage in your model ===")
    print("You can now pass autotune_preset as a parameter:")
    print("main(vocals_file='audio.wav', autotune_preset='natural')")

if __name__ == "__main__":
    example_usage() 