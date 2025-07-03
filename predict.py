import os
import tempfile
import librosa
from cog import BasePredictor, Input, Path
import requests
import numpy as np
from scipy.signal import convolve, windows # <-- Import for smoothing

# Import the pipeline steps
from pipeline.step_01_preprocess import preprocess_audio
from pipeline.step_02_pitch_detection import detect_pitch
from pipeline.step_03_correction import create_target_pitch_contour
from pipeline.step_04_resynthesis import resynthesize_with_rubberband
from pipeline.step_05_harmonize import generate_harmony_contours
from utils.plotter import create_plot

class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def get_local_path(self, file: Path) -> str:
        # This function remains the same
        file_str = str(file)
        if file_str.startswith("https:/") and not file_str.startswith("https://"):
            file_str = file_str.replace("https:/", "https://", 1)
        if os.path.exists(file_str):
            return file_str
        if file_str.startswith("http"):
            response = requests.get(file_str)
            response.raise_for_status()
            suffix = os.path.splitext(file_str)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                return tmp.name
        raise FileNotFoundError(f"File not found: {file_str}")

    def predict(
        self,
        audio_file: Path = Input(description="Audio file to process."),
        effect_mode: str = Input(
            description="Choose between simple pitch correction or a creative harmonizer effect.",
            default="harmonizer",
            choices=["corrective", "harmonizer"]
        ),
        harmony_preset: str = Input(
            description="[Harmonizer Mode Only] The harmony to generate.",
            default="major_third",
            choices=["major_third", "minor_third", "fifth", "octave"]
        ),
        scale: str = Input(
            description="Musical key for pitch correction.",
            default="C:maj",
            choices=["closest", "A:maj", "A:min", "Bb:maj", "Bb:min", "B:maj", "B:min", "C:maj", "C:min", "Db:maj", "Db:min", "D:maj", "D:min", "Eb:maj", "Eb:min", "E:maj", "E:min", "F:maj", "F:min", "Gb:maj", "Gb:min", "G:maj", "G:min", "Ab:maj", "Ab:min"],
        ),
        glide_speed_ms: int = Input(
            description="The time it takes to glide between notes. Lower is more robotic.",
            default=15, ge=0, le=100
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        )
    ) -> list[Path]:
        """Runs the complete auto-tune pipeline."""
        
        print("--- Starting Auto-Tune Pipeline ---")
        
        local_audio_path = self.get_local_path(audio_file)
        y, sr = librosa.load(local_audio_path, sr=None, mono=True)
        
        print("Step 1/6: Pre-processing audio...")
        y_clean = preprocess_audio(y, sr)
        
        print("Step 2/6: Detecting pitch...")
        time_steps, original_f0 = detect_pitch(y_clean, sr)

        print("Step 3/6: Applying smooth noise gate...")
        hop_length = int(sr / 100)
        voiced_mask = ~np.isnan(original_f0)
        full_voiced_mask = np.repeat(voiced_mask, hop_length)
        len_diff = len(y_clean) - len(full_voiced_mask)
        if len_diff > 0:
            full_voiced_mask = np.pad(full_voiced_mask, (0, len_diff), 'constant', constant_values=False)
        else:
            full_voiced_mask = full_voiced_mask[:len(y_clean)]
        
        # --- NEW SMOOTHING LOGIC ---
        # Create a smoothing window (e.g., 50ms) to create fade-ins/outs
        smoothing_window_size = int(sr * 0.05)
        smoothing_window = windows.hann(smoothing_window_size)
        smooth_mask = convolve(full_voiced_mask, smoothing_window, mode='same') / np.sum(smoothing_window)
        y_gated_clean = y_clean * smooth_mask

        print("Step 4/6: Generating target pitch...")
        corrected_f0 = create_target_pitch_contour(original_f0, time_steps, sr, scale, glide_speed_ms)
        
        final_audio = None
        
        if effect_mode == "corrective":
            print("Step 5/6: Resynthesizing corrected vocal...")
            final_audio = resynthesize_with_rubberband(y_gated_clean, sr, original_f0, corrected_f0, time_steps)
        
        elif effect_mode == "harmonizer":
            print("Step 5/6: Generating and mixing harmony parts...")
            harmony_contours = generate_harmony_contours(corrected_f0, sr, harmony_preset)
            
            # For harmonizer, use the original audio as lead and only shift the harmony
            lead_vocal = y_gated_clean  # Use original audio for lead
            harmony_vocal = resynthesize_with_rubberband(y_gated_clean, sr, original_f0, harmony_contours[1], time_steps)
            
            if len(lead_vocal) != len(harmony_vocal):
                min_len = min(len(lead_vocal), len(harmony_vocal))
                lead_vocal = lead_vocal[:min_len]
                harmony_vocal = harmony_vocal[:min_len]
            
            # Simple mixing without delay - just blend the tracks
            final_audio = (lead_vocal * 1.0) + (harmony_vocal * 0.3)
            
        print("Step 6/6: Normalizing final output...")
        
        # Calculate RMS of original audio for loudness matching
        original_rms = np.sqrt(np.mean(y_clean**2))
        output_rms = np.sqrt(np.mean(final_audio**2))
        
        # Match loudness to original
        if output_rms > 0:
            final_audio = final_audio * (original_rms / output_rms)
        
        # Peak normalization to prevent clipping
        peak_level = np.max(np.abs(final_audio))
        if peak_level > 0:
            y_normalized = final_audio / peak_level * 0.94
        else:
            y_normalized = final_audio

        # Save to current directory instead of temp directory
        temp_wav_path = "temp_corrected.wav"
        final_audio_path = f"output.{output_format}"
        plot_output_path = "pitch_correction.png"
        
        import soundfile as sf
        sf.write(temp_wav_path, y_normalized, sr)

        if output_format == "mp3":
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", temp_wav_path, final_audio_path], check=True, capture_output=True)
            # Clean up temp file
            os.remove(temp_wav_path)
        else:
            os.rename(temp_wav_path, final_audio_path)

        create_plot(time_steps, original_f0, corrected_f0, plot_output_path)

        print("--- Pipeline Complete ---")
        print(f"Output saved to: {final_audio_path}")
        print(f"Plot saved to: {plot_output_path}")
        return [Path(final_audio_path), Path(plot_output_path)]

if __name__ == "__main__":
    p = Predictor()
    p.setup()
    
    outputs = p.predict(
        audio_file=Path("https://storage.googleapis.com/riffgen/audio/ddf3a0b9-efee-42e3-9838-9e6ba7e0ac0d.m4a"),
        effect_mode="harmonizer",
        harmony_preset="major_third",
        scale="A:min",
        glide_speed_ms=25, # A slightly slower glide can also sound smoother
        output_format="wav",
    )
    
    print("Output files generated:")
    for file_path in outputs:
        print(f"- {file_path}")