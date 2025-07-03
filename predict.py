# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import tempfile
import requests
import librosa
import numpy as np
from cog import BasePredictor, Input, Path

# Make sure your corrected utils file is named 'utils.py'
from utils import main

def estimate_key(filepath: str) -> str:
    """Estimates the key of an audio file, returning a string like 'C:maj'."""
    y, sr = librosa.load(filepath, sr=None)
    # Using the original audio is much faster than running HPSS first
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    key_corrs_maj = librosa.key.key_correlation(chroma, 'major')
    key_corrs_min = librosa.key.key_correlation(chroma, 'minor')

    if np.max(key_corrs_maj) > np.max(key_corrs_min):
        key_idx = np.argmax(key_corrs_maj)
        return f"{notes[key_idx]}:maj"
    else:
        key_idx = np.argmax(key_corrs_min)
        return f"{notes[key_idx]}:min"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def get_local_path(self, audio_file: Path) -> str:
        if os.path.exists(str(audio_file)):
            return str(audio_file)
        if str(audio_file).startswith("http"):
            response = requests.get(str(audio_file))
            response.raise_for_status()
            suffix = os.path.splitext(str(audio_file))[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                return tmp.name
        raise ValueError("audio_file must be a local path or a URL")

    def predict(
        self,
        audio_file: Path = Input(description="URL or path to the audio file to be pitch-corrected"),
        scale: str = Input(
            description="Musical key for correction. 'auto' will detect the key, 'closest' uses chromatic correction.",
            default="auto",
            choices=["auto", "closest", "A:maj", "A:min", "Bb:maj", "Bb:min", "B:maj", "B:min", "C:maj", "C:min", "Db:maj", "Db:min", "D:maj", "D:min", "Eb:maj", "Eb:min", "E:maj", "E:min", "F:maj", "F:min", "Gb:maj", "Gb:min", "G:maj", "G:min", "Ab:maj", "Ab:min"],
        ),
        correction_strength: float = Input(
            description="Strength of pitch correction (0.0 to 1.0). Higher values = stronger correction.",
            default=0.7, ge=0.0, le=1.0,
        ),
        adaptive_strength: bool = Input(
            description="Adapt correction strength based on voice confidence.",
            default=True,
        ),
        smooth_transitions: bool = Input(
            description="Smooth transitions between voiced/unvoiced segments.",
            default=True,
        ),
        plot: bool = Input(
            description="Generate a pitch correction visualization plot.",
            default=False,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        filepath = self.get_local_path(audio_file)

        # Determine the actual scale to use
        actual_scale = scale
        correction_method = "scale" if scale != "closest" else "closest"
        if scale == "auto":
            try:
                print("🎤 Automatically detecting key...")
                actual_scale = estimate_key(filepath)
                correction_method = "scale"
                print(f"✅ Detected key: {actual_scale}")
            except Exception as e:
                print(f"⚠️ Key detection failed: {e}. Falling back to chromatic correction.")
                actual_scale = "closest"
                correction_method = "closest"

        # Call the main processing function
        main(
            vocals_file=filepath,
            correction_method=correction_method,
            scale_key=actual_scale,
            correction_strength=correction_strength,
            adaptive_strength=adaptive_strength,
            smooth_transitions=smooth_transitions,
            plot=plot
        )

        import pathlib
        input_path = pathlib.Path(filepath)
        output_wav = input_path.parent / f"{input_path.stem}_pitch_corrected.wav"
        
        if not output_wav.exists():
            raise FileNotFoundError(f"Expected output file not found: {output_wav}")

        if output_format == "mp3":
            output_path_str = "output.mp3"
            if os.path.exists(output_path_str):
                os.remove(output_path_str)
            subprocess.run(["ffmpeg", "-y", "-i", str(output_wav), output_path_str], check=True, capture_output=True)
            os.remove(str(output_wav))
        else:
            output_path_str = "output.wav"
            if os.path.exists(output_path_str):
                os.remove(output_path_str)
            os.rename(str(output_wav), output_path_str)

        return Path(output_path_str)

if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        # audio_file="https://storage.googleapis.com/riffgen/audio/f9398d3d-1d48-48d2-adc2-2c28b8caabf8.m4a",
        # audio_file="https://storage.googleapis.com/riffgen/audio/a12debeb-f3c8-454b-88cd-d87889e51ddd.m4a",
        audio_file="https://storage.googleapis.com/riffgen/audio/33e1f7fd-3a65-4e8c-a650-013f05b2ae07.m4a",
        scale="auto",
        correction_strength=1,
        adaptive_strength=True,
        smooth_transitions=True,
        plot=True,
        output_format="wav",
    )
    print("Output file:", out)