from functools import partial
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
import psola
import numbers


SEMITONES_IN_OCTAVE = 12


def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
    # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
    # would be incorrectly assigned.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees


def closest_pitch(f0, correction_strength=1.0):
    """Round the given pitch values to the nearest MIDI note numbers with controllable strength"""
    if isinstance(f0, numbers.Number):
        if np.isnan(f0):
            return np.nan
        midi_note = librosa.hz_to_midi(f0)
        target_midi = np.around(midi_note)
        corrected_midi = midi_note + (target_midi - midi_note) * correction_strength
        return librosa.midi_to_hz(corrected_midi)
    else:
        midi_note = librosa.hz_to_midi(f0)
        target_midi = np.around(midi_note)
        corrected_midi = midi_note + (target_midi - midi_note) * correction_strength
        nan_indices = np.isnan(f0)
        corrected_midi[nan_indices] = np.nan
        return librosa.midi_to_hz(corrected_midi)


def closest_pitch_from_scale(f0, scale, correction_strength=1.0):
    """Return the pitch closest to f0 that belongs to the given scale with controllable strength"""
    if isinstance(f0, numbers.Number):
        if np.isnan(f0):
            return np.nan
        degrees = degrees_from(scale)
        midi_note = librosa.hz_to_midi(f0)
        degree = midi_note % SEMITONES_IN_OCTAVE
        degree_id = np.argmin(np.abs(degrees - degree))
        degree_difference = degree - degrees[degree_id]
        target_midi = midi_note - degree_difference
        corrected_midi = midi_note + (target_midi - midi_note) * correction_strength
        return librosa.midi_to_hz(corrected_midi)
    else:
        degrees = degrees_from(scale)
        midi_note = librosa.hz_to_midi(f0)
        degree = midi_note % SEMITONES_IN_OCTAVE
        degree_id = np.argmin(np.abs(degrees - degree), axis=0)
        degree_difference = degree - degrees[degree_id]
        target_midi = midi_note - degree_difference
        corrected_midi = midi_note + (target_midi - midi_note) * correction_strength
        nan_indices = np.isnan(f0)
        corrected_midi[nan_indices] = np.nan
        return librosa.midi_to_hz(corrected_midi)


def aclosest_pitch_from_scale(f0, scale, correction_strength=1.0):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale, correction_strength)
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def smooth_pitch_transitions(f0, voiced_flag, window_size=5):
    """Smooth pitch transitions between voiced and unvoiced segments"""
    smoothed_f0 = f0.copy()
    
    # Find voiced/unvoiced boundaries
    voiced_changes = np.diff(voiced_flag.astype(int))
    voiced_starts = np.where(voiced_changes == 1)[0]
    voiced_ends = np.where(voiced_changes == -1)[0]
    
    # Smooth transitions at voiced segment boundaries
    for start in voiced_starts:
        if start > 0:
            # Smooth the start of voiced segments
            start_idx = max(0, start - window_size)
            end_idx = min(len(f0), start + window_size)
            
            # Create a smooth transition from the previous pitch to the voiced pitch
            if not np.isnan(f0[start]) and not np.isnan(f0[start - 1]):
                transition = np.linspace(f0[start - 1], f0[start], end_idx - start_idx)
                smoothed_f0[start_idx:end_idx] = transition
    
    for end in voiced_ends:
        if end < len(f0) - 1:
            # Smooth the end of voiced segments
            start_idx = max(0, end - window_size)
            end_idx = min(len(f0), end + window_size)
            
            # Create a smooth transition from the voiced pitch to the next pitch
            if not np.isnan(f0[end]) and not np.isnan(f0[end + 1]):
                transition = np.linspace(f0[end], f0[end + 1], end_idx - start_idx)
                smoothed_f0[start_idx:end_idx] = transition
    
    return smoothed_f0


def adaptive_correction_strength(f0, voiced_probabilities, base_strength=0.8):
    """Adapt correction strength based on voice confidence and pitch stability"""
    # Higher confidence = stronger correction
    confidence_factor = voiced_probabilities
    
    # Calculate pitch stability (how much pitch changes between frames)
    pitch_diff = np.abs(np.diff(f0, prepend=f0[0]))
    pitch_stability = 1.0 / (1.0 + pitch_diff / 50.0)  # Normalize around 50Hz difference
    
    # Combine factors
    adaptive_strength = base_strength * confidence_factor * pitch_stability
    
    # Clamp to reasonable range
    adaptive_strength = np.clip(adaptive_strength, 0.1, 1.0)
    
    return adaptive_strength


def autotune(audio, sr, correction_function, plot=False, 
             correction_strength=0.8, 
             adaptive_strength=True, smooth_transitions=True):
    # Adaptive frame parameters based on audio characteristics
    frame_length = min(2048, len(audio) // 4)  # Adaptive frame length
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm (no unsupported args)
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    # Apply adaptive correction strength if enabled
    if adaptive_strength:
        strength_array = adaptive_correction_strength(f0, voiced_probabilities, correction_strength)
        corrected_f0 = np.zeros_like(f0)
        # Handle partial functions
        func_name = getattr(correction_function, '__name__', None)
        if func_name is None and hasattr(correction_function, 'func'):
            func_name = getattr(correction_function.func, '__name__', None)
        for i in range(len(f0)):
            if func_name == 'closest_pitch':
                corrected_f0[i] = closest_pitch(f0[i], strength_array[i])
            else:
                corrected_f0[i] = correction_function(f0[i])
    else:
        corrected_f0 = correction_function(f0)

    if smooth_transitions:
        corrected_f0 = smooth_pitch_transitions(corrected_f0, voiced_flag)

    if plot:
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [M:SS]')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    # Pitch-shifting using the PSOLA algorithm
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, 
                       fmin=fmin, fmax=fmax)


def main(
    vocals_file,
    plot=False,
    correction_method="closest",
    scale=None,
    correction_strength=0.8,
    adaptive_strength=True,
    smooth_transitions=True
):
    """Run autotune-like pitch correction on the given audio file.

    Args:
        vocals_file (str): Filepath to the audio file to be pitch-corrected.
        plot (bool, optional): Whether to plot the results. Defaults to False.
        correction_method (str, optional): The pitch correction method to use. Defaults to `"closest"`. If set to "closest", the pitch will be rounded to the nearest MIDI note.
            If set to "scale", the pitch will be rounded to the nearest note in the given `scale`.
        scale (str, optional): The scale to use for pitch correction. ex. `"C:min"` / `"A:maj"`. Defaults to None.
        correction_strength (float, optional): Strength of pitch correction (0.0 to 1.0). Defaults to 0.8.
        adaptive_strength (bool, optional): Whether to adapt correction strength based on voice confidence. Defaults to True.
        smooth_transitions (bool, optional): Whether to smooth transitions between voiced/unvoiced segments. Defaults to True.
    """    
    
    filepath = Path(vocals_file)

    # Load the audio file.
    y, sr = librosa.load(str(filepath), sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Pick the pitch adjustment strategy according to the arguments.
    if correction_method == 'closest':
        correction_function = closest_pitch
    else:
        correction_function = partial(aclosest_pitch_from_scale, scale=scale)

    # Perform the auto-tuning.
    pitch_corrected_y = autotune(y, sr, correction_function, plot, 
                                correction_strength, 
                                adaptive_strength, smooth_transitions)

    # Write the corrected audio to an output file.
    # Force output to .wav for compatibility
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected.wav')
    sf.write(str(filepath), pitch_corrected_y, sr)
    return pitch_corrected_y


if __name__=='__main__':
    from fire import Fire
    Fire(main)
