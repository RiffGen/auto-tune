import numpy as np
import librosa
from scipy.signal import windows, convolve

# --- SIMPLE HARMONY PRESETS ---
# Using very small intervals for natural sound
HARMONY_PRESETS = {
    "major_third": 2.0,   # Just 2 semitones up (much smaller than before)
    "minor_third": 1.5,   # 1.5 semitones up
    "fifth": 3.0,         # 3 semitones up (reduced from 7)
    "octave": 7.0,        # Octave (kept as is)
}

def generate_harmony_contours(base_f0: 'np.ndarray', sr: int, preset: str) -> 'list[np.ndarray]':
    """
    Generates a pitch contour for the lead vocal and one harmony line.
    Uses a much simpler approach to avoid artifacts.
    """
    if preset not in HARMONY_PRESETS:
        raise ValueError(f"Unknown harmony preset: {preset}. Available presets are: {list(HARMONY_PRESETS.keys())}")

    # Keep the lead vocal mostly unchanged (just slight correction)
    lead_f0 = np.copy(base_f0)
    
    # Create a simple harmony by shifting the pitch contour
    semitone_shift = HARMONY_PRESETS[preset]
    
    # Convert to MIDI, shift, convert back
    base_midi = librosa.hz_to_midi(base_f0)
    harmony_midi = base_midi + semitone_shift
    harmony_f0 = librosa.midi_to_hz(harmony_midi)
    
    # Clip to reasonable vocal range to avoid artifacts
    harmony_f0 = np.clip(harmony_f0, 80, 800)
    
    return [lead_f0, harmony_f0]