import numpy as np
import subprocess
import tempfile
import soundfile as sf
import os

def resynthesize_with_rubberband(y: 'np.ndarray', sr: int, original_f0: 'np.ndarray', target_f0: 'np.ndarray', time_steps: 'np.ndarray') -> 'np.ndarray':
    """
    Uses the Rubber Band library with a frequency multiplier map, providing the
    correct sample frame number format.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        input_path = os.path.join(tempdir, "input.wav")
        output_path = os.path.join(tempdir, "output.wav")
        freq_map_path = os.path.join(tempdir, "freq_map.txt")
        
        sf.write(input_path, y, sr)
        
        has_valid_pitch = False
        with open(freq_map_path, 'w') as f:
            for i, t in enumerate(time_steps):
                if not np.isnan(original_f0[i]) and original_f0[i] > 0 and not np.isnan(target_f0[i]) and target_f0[i] > 0:
                    
                    # --- THE FINAL FIX ---
                    # Convert the time in seconds (t) to an integer sample frame number.
                    frame_number = int(t * sr)
                    
                    multiplier = target_f0[i] / original_f0[i]
                    f.write(f"{frame_number} {multiplier}\n")
                    has_valid_pitch = True
        
        if not has_valid_pitch:
            print("Warning: No valid pitch points found to correct. Returning original audio.")
            return y
        
        # Debug: Print the frequency map contents
        print(f"DEBUG: Frequency map file contents:")
        with open(freq_map_path, 'r') as f:
            lines = f.readlines()
            print(f"Total lines: {len(lines)}")
            for i, line in enumerate(lines[:10]):
                print(f"  {i+1}: {repr(line)}")
        
        command = [
            "rubberband",
            "--freqmap", freq_map_path,
            "--time", "1",
            "--formant",  # Preserve formants for more natural sound
            "--smoothing",  # Enable smoothing to reduce artifacts
            "-q",
            input_path,
            output_path
        ]
        
        print(f"DEBUG: Rubber Band command: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("ERROR: rubberband command failed.")
            print(f"Exit Code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise e

        y_corrected, _ = sf.read(output_path)
        
        return y_corrected