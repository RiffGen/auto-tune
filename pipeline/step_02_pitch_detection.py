import crepe
import numpy as np
from scipy.signal import medfilt

CONF_THRESHOLD = 0.6  # Confidence threshold to consider a pitch "real"

def detect_pitch(y: 'np.ndarray', sr: int) -> 'tuple[np.ndarray, np.ndarray]':
    """
    Uses the CREPE model to detect pitch and cleans up the results.
    """
    time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True, model_capacity="tiny")
    
    # Set frequency to NaN where confidence is below the threshold
    frequency[confidence < CONF_THRESHOLD] = np.nan
    
    # Optional: Apply a median filter to remove single-frame pitch errors (octave jumps)
    frequency = medfilt(frequency, kernel_size=3)
    
    return time, frequency