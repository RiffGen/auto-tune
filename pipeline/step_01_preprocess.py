import noisereduce as nr
import librosa

def preprocess_audio(y: 'np.ndarray', sr: int) -> 'np.ndarray':
    """
    Cleans up the raw audio signal.
    """
    # Use a basic noise reduction. You can get more advanced here.
    y_clean = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.9)
    
    # Optional: Apply a high-pass filter to remove low-frequency rumble
    y_clean = librosa.effects.preemphasis(y_clean)
    
    return y_clean