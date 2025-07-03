#!/usr/bin/env python3
"""
Test script for experimenting with different auto-tune settings
"""

import os
import requests
import pathlib
from pitch_correction_utils import main

def download_audio(url, filename="input_audio.m4a"):
    """Download audio file from URL"""
    print(f"Downloading audio from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded: {filename}")
        return filename
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None

def test_different_settings(input_file):
    """Test various auto-tune settings and save results"""
    
    # Create output directory
    output_dir = "autotune_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations with clear parameter descriptions
    configs = [
        {
            "name": "01_natural_subtle",
            "description": "Very subtle correction - maintains natural voice",
            "correction_strength": 0.2,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "02_gentle_correction",
            "description": "Gentle correction - slight pitch improvement",
            "correction_strength": 0.4,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "03_balanced_correction",
            "description": "Balanced correction - good pitch accuracy with natural sound",
            "correction_strength": 0.6,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "04_strong_correction",
            "description": "Strong correction - noticeable but not robotic",
            "correction_strength": 0.8,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "05_classic_autotune",
            "description": "Classic autotune sound - full correction",
            "correction_strength": 1.0,
            "adaptive_strength": False,
            "smooth_transitions": False
        },
        {
            "name": "06_scale_c_major",
            "description": "Scale-based correction in C major",
            "correction_method": "scale",
            "scale": "C:maj",
            "correction_strength": 0.7,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "07_scale_a_minor",
            "description": "Scale-based correction in A minor",
            "correction_method": "scale",
            "scale": "A:min",
            "correction_strength": 0.7,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "08_no_smoothing",
            "description": "Strong correction without transition smoothing",
            "correction_strength": 0.8,
            "adaptive_strength": True,
            "smooth_transitions": False
        },
        {
            "name": "09_ultra_subtle",
            "description": "Ultra subtle - barely noticeable correction",
            "correction_strength": 0.1,
            "adaptive_strength": True,
            "smooth_transitions": True
        },
        {
            "name": "10_preset_natural",
            "description": "Using preset: natural (demonstrates preset functionality)",
            "autotune_preset": "natural"
        },
        {
            "name": "11_preset_classic",
            "description": "Using preset: classic (demonstrates preset functionality)",
            "autotune_preset": "classic"
        },
        {
            "name": "12_preset_c_major",
            "description": "Using preset: c_major (demonstrates preset functionality)",
            "autotune_preset": "c_major"
        }
    ]
    
    print(f"Testing auto-tune settings on: {input_file}")
    print("=" * 80)
    
    # Create a summary file
    summary_file = pathlib.Path(output_dir) / "test_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("AUTO-TUNE TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {input_file}\n\n")
        f.write("Test Configurations:\n")
        f.write("-" * 30 + "\n")
        
        for i, config in enumerate(configs, 1):
            f.write(f"{i:2d}. {config['name']}\n")
            f.write(f"    Description: {config['description']}\n")
            if "autotune_preset" in config:
                f.write(f"    Preset: {config['autotune_preset']}\n")
            else:
                f.write(f"    Correction Strength: {config['correction_strength']}\n")
                f.write(f"    Adaptive Strength: {config['adaptive_strength']}\n")
                f.write(f"    Smooth Transitions: {config['smooth_transitions']}\n")
                if 'scale' in config:
                    f.write(f"    Scale: {config['scale']}\n")
            f.write("\n")
    
    print(f"Created test summary: {summary_file}")
    print("\nRunning tests...\n")
    
    successful_tests = []
    failed_tests = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        print(f"  {config['description']}")
        
        # Run auto-tune with current config
        try:
            # Handle preset vs individual parameters
            if "autotune_preset" in config:
                result = main(
                    vocals_file=input_file,
                    plot=True,
                    autotune_preset=config["autotune_preset"]
                )
            else:
                result = main(
                    vocals_file=input_file,
                    plot=True,
                    correction_method=config.get("correction_method", "closest"),
                    scale=config.get("scale"),
                    correction_strength=config["correction_strength"],
                    adaptive_strength=config["adaptive_strength"],
                    smooth_transitions=config["smooth_transitions"]
                )
            
            # Rename output file to include config name
            input_path = pathlib.Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_pitch_corrected.wav"
            new_output_file = pathlib.Path(output_dir) / f"{config['name']}.wav"
            
            if output_file.exists():
                output_file.rename(new_output_file)
                print(f"  ✓ Audio: {new_output_file}")
            
            # Rename plot file
            plot_file = pathlib.Path("pitch_correction.png")
            if plot_file.exists():
                new_plot_file = pathlib.Path(output_dir) / f"{config['name']}_plot.png"
                plot_file.rename(new_plot_file)
                print(f"  ✓ Plot: {new_plot_file}")
            
            successful_tests.append(config['name'])
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_tests.append(config['name'])
        
        print()
    
    # Create a comparison guide
    comparison_file = pathlib.Path(output_dir) / "comparison_guide.txt"
    with open(comparison_file, 'w') as f:
        f.write("AUTO-TUNE COMPARISON GUIDE\n")
        f.write("=" * 30 + "\n\n")
        f.write("Listen to these files in order to compare:\n\n")
        
        for config in configs:
            if config['name'] in successful_tests:
                f.write(f"• {config['name']}.wav\n")
                f.write(f"  {config['description']}\n\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        f.write("• For natural sound: Try 01_natural_subtle or 02_gentle_correction\n")
        f.write("• For pop music: Try 03_balanced_correction or 04_strong_correction\n")
        f.write("• For classic autotune effect: Use 05_classic_autotune\n")
        f.write("• For specific scales: Use 06_scale_c_major or 07_scale_a_minor\n")
        f.write("• For ultra subtle: Use 09_ultra_subtle\n")
    
    print("=" * 80)
    print(f"✓ Completed {len(successful_tests)} successful tests")
    if failed_tests:
        print(f"✗ {len(failed_tests)} tests failed: {', '.join(failed_tests)}")
    
    print(f"\n📁 Check the '{output_dir}' folder for:")
    print(f"   • Audio files with different settings")
    print(f"   • Pitch correction plots")
    print(f"   • test_summary.txt - detailed parameters")
    print(f"   • comparison_guide.txt - listening recommendations")
    
    return successful_tests, failed_tests

if __name__ == "__main__":
    import sys
    
    # Default URL if no arguments provided
    url = "https://storage.googleapis.com/riffgen/audio/f9398d3d-1d48-48d2-adc2-2c28b8caabf8.m4a"
    input_file = "input_audio.m4a"
    
    if len(sys.argv) > 1:
        # If argument provided, treat as local file
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Error: Local file '{input_file}' not found!")
            sys.exit(1)
    else:
        # Download from URL
        input_file = download_audio(url)
        if not input_file:
            sys.exit(1)
    
    test_different_settings(input_file) 