# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import tempfile
import requests

from cog import BasePredictor, Input, Path

from old.utils_old import main


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def get_local_path(self, audio_file):
        # If it's already a local file, just return the path
        if os.path.exists(str(audio_file)):
            return str(audio_file)
        # If it's a URL, download it to a temp file
        if str(audio_file).startswith("http://") or str(audio_file).startswith("https://"):
            response = requests.get(audio_file)
            response.raise_for_status()
            suffix = os.path.splitext(audio_file)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                return tmp.name
        raise ValueError("audio_file must be a local path or a URL")

    def predict(
        self,
        audio_file: Path = Input(description="URL or path to the audio file to be pitch-corrected"),
        autotune_style: str = Input(
            description="Auto-tune style/intensity preset. Changing this will update the defaults for correction strength, adaptive, and smoothing, but you can override them below.",
            default="balanced",
            choices=["ultra_subtle", "natural", "gentle", "balanced", "strong", "classic"],
        ),
        scale: str = Input(
            description="Musical key/scale to use for pitch correction. If set to 'closest', will use chromatic correction.",
            default="closest",
            choices=[
                "closest", "A:maj", "A:min", "Bb:maj", "Bb:min", "B:maj", "B:min", "C:maj", "C:min", "Db:maj", "Db:min", "D:maj", "D:min", "Eb:maj", "Eb:min", "E:maj", "E:min", "F:maj", "F:min", "Gb:maj", "Gb:min", "G:maj", "G:min", "Ab:maj", "Ab:min"
            ],
        ),
        correction_strength: float = Input(
            description="Strength of pitch correction (0.0 to 1.0). Defaults to the style preset, but you can override.",
            default=0.8,
        ),
        adaptive_strength: bool = Input(
            description="Whether to adapt correction strength based on voice confidence. Defaults to the style preset, but you can override.",
            default=True,
        ),
        smooth_transitions: bool = Input(
            description="Whether to smooth transitions between voiced/unvoiced segments. Defaults to the style preset, but you can override.",
            default=True,
        ),
        plot: bool = Input(
            description="Whether to generate a pitch correction visualization plot",
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

        # Style preset defaults
        style_defaults = {
            "ultra_subtle": {"correction_strength": 0.1, "adaptive_strength": True, "smooth_transitions": True},
            "natural":      {"correction_strength": 0.2, "adaptive_strength": True, "smooth_transitions": True},
            "gentle":       {"correction_strength": 0.4, "adaptive_strength": True, "smooth_transitions": True},
            "balanced":     {"correction_strength": 0.6, "adaptive_strength": True, "smooth_transitions": True},
            "strong":       {"correction_strength": 0.8, "adaptive_strength": True, "smooth_transitions": True},
            "classic":      {"correction_strength": 1.0, "adaptive_strength": False, "smooth_transitions": False},
        }
        defaults = style_defaults.get(autotune_style, style_defaults["balanced"])

        # Use user overrides if they differ from the preset
        cs = correction_strength if correction_strength != defaults["correction_strength"] else defaults["correction_strength"]
        ad = adaptive_strength if adaptive_strength != defaults["adaptive_strength"] else defaults["adaptive_strength"]
        sm = smooth_transitions if smooth_transitions != defaults["smooth_transitions"] else defaults["smooth_transitions"]

        # Always use the selected scale
        pitch_corrected_y = main(
            vocals_file=filepath,
            correction_method="scale" if scale != "closest" else "closest",
            scale=scale,
            correction_strength=cs,
            adaptive_strength=ad,
            smooth_transitions=sm,
            plot=plot
        )

        # The main function already saves as .wav, so we need to find that file
        import pathlib
        input_path = pathlib.Path(filepath)
        output_wav = input_path.parent / f"{input_path.stem}_pitch_corrected.wav"
        
        if output_format == "mp3":
            mp3_path = "output.mp3"
            if os.path.isfile(mp3_path):
                os.remove(mp3_path)
            subprocess.call(["ffmpeg", "-y", "-i", str(output_wav), mp3_path])
            os.remove(str(output_wav))
            path = mp3_path
        else:
            # For wav output, rename to output.wav
            wav_path = "output.wav"
            if os.path.isfile(wav_path):
                os.remove(wav_path)
            os.rename(str(output_wav), wav_path)
            path = wav_path

        return Path(path)


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        # audio_file="https://storage.googleapis.com/riffgen/audio/f9398d3d-1d48-48d2-adc2-2c28b8caabf8.m4a",
        audio_file="https://storage.googleapis.com/riffgen/audio/a12debeb-f3c8-454b-88cd-d87889e51ddd.m4a",
        autotune_style="balanced",
        scale="closest",
        # scale="A:maj",
        correction_strength=0.8,
        adaptive_strength=True,
        smooth_transitions=True,
        plot=False,
        output_format="mp3",
    )
    print("Output file:", out)