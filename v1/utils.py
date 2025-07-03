from functools import partial
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
from scipy.ndimage import binary_opening, binary_closing
import pandas as pd
import psola

SEMITONES_IN_OCTAVE = 12

def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale."""
    degrees = librosa.key_to_degrees(scale)
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees

def find_target_pitch(f0, correction_function):
    """Vectorized function to find the ideal target pitch for each frame."""
    midi_note = librosa.hz_to_midi(f0)
    
    if correction_function == closest_pitch:
        target_midi = np.around(midi_note)
        return librosa.midi_to_hz(target_midi)

    scale = correction_function.keywords.get('scale')
    if not scale:
        target_midi = np.around(midi_note)
        return librosa.midi_to_hz(target_midi)

    degrees = degrees_from(scale)
    pitch_class = midi_note % SEMITONES_IN_OCTAVE
    diffs = np.abs(pitch_class[:, np.newaxis] - degrees)
    diffs = np.minimum(diffs, 12 - diffs)
    closest_degree_indices = np.argmin(diffs, axis=1)
    target_degrees = degrees[closest_degree_indices]
    degree_diff = pitch_class - target_degrees
    target_midi = midi_note - degree_diff
    target_midi[np.isnan(f0)] = np.nan
    
    return librosa.midi_to_hz(target_midi)

def correct_octave_errors(f0, max_jump_semitones=10):
    """Corrects large, sudden jumps in pitch likely caused by octave errors."""
    if np.all(np.isnan(f0)):
        return f0
    midi_f0 = librosa.hz_to_midi(f0)
    diffs = np.diff(midi_f0, prepend=midi_f0[0])
    jump_indices = np.where(np.abs(diffs) > max_jump_semitones)[0]
    
    for i in jump_indices:
        if i == 0: continue
        if np.isnan(midi_f0[i]) or np.isnan(midi_f0[i-1]): continue
        direction = -1 if diffs[i] > 0 else 1
        midi_f0[i:] += direction * SEMITONES_IN_OCTAVE
        
    return librosa.midi_to_hz(midi_f0)

def contour_pitch(f0, target_f0, strength):
    """
    Shifts the original pitch contour towards the target, preserving vibrato.
    This is the key function for a natural sound.
    """
    # Find the center of the original pitch for each sustained note
    midi_f0 = librosa.hz_to_midi(f0)
    smoothed_midi = sig.medfilt(midi_f0, kernel_size=11) # Heavily smooth to find stable segments
    rounded_midi = np.round(smoothed_midi)
    
    # Find segments where the intended note is stable
    note_changes = np.where(np.diff(rounded_midi) != 0)[0]
    segment_starts = np.insert(note_changes + 1, 0, 0)
    
    final_corrected_f0 = np.copy(f0)

    for i in range(len(segment_starts)):
        start = segment_starts[i]
        end = segment_starts[i+1] if i + 1 < len(segment_starts) else len(f0)
        
        if start >= end:
            continue
            
        # Get the single, stable target for this whole segment
        segment_target_hz = np.nanmedian(target_f0[start:end])
        if np.isnan(segment_target_hz):
            continue

        # Get the average original pitch for the segment
        segment_original_hz = np.nanmean(f0[start:end])
        if np.isnan(segment_original_hz):
            continue

        # Calculate the required shift and apply it with the desired strength
        pitch_shift = segment_target_hz - segment_original_hz
        correction_to_apply = pitch_shift * strength

        # Apply the same correction offset to the entire original segment
        final_corrected_f0[start:end] = f0[start:end] + correction_to_apply
            
    return final_corrected_f0


def autotune(audio, sr, correction_function, plot=False,
             correction_strength=0.8,
             adaptive_strength=False, # Adaptive strength is complex with this new method, disable for now
             smooth_transitions=True):

    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    f0, voiced_flag, voiced_probabilities = librosa.pyin(
        audio, frame_length=frame_length, hop_length=hop_length, sr=sr, fmin=fmin, fmax=fmax
    )
    
    f0 = correct_octave_errors(f0)
    
    structure = np.ones(5)
    voiced_flag = binary_opening(voiced_flag, structure=structure)
    voiced_flag = binary_closing(voiced_flag, structure=structure)
    
    f0[~voiced_flag] = np.nan
    f0 = pd.Series(f0).interpolate().to_numpy()
    f0 = np.nan_to_num(f0)

    target_f0 = find_target_pitch(f0, correction_function)

    # **NEW: Use the contouring method for the final pitch correction.**
    corrected_f0 = contour_pitch(f0, target_f0, correction_strength)
    corrected_f0[~voiced_flag] = 0 # Ensure unvoiced sections are silent

    if plot:
        time_points = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_points, f0, label='Original Pitch (Cleaned)', color='cyan', linewidth=2, alpha=0.8)
        ax.plot(time_points, corrected_f0, label='Final Corrected Pitch', color='orange', linewidth=2)
        ax.legend()
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def closest_pitch(): pass
def aclosest_pitch_from_scale(): pass

def main(vocals_file, plot=False, correction_method="closest", scale_key=None,
         correction_strength=0.8, adaptive_strength=True, smooth_transitions=True, **kwargs):
    filepath = Path(vocals_file)
    y, sr = librosa.load(str(filepath), sr=None, mono=True)

    if correction_method == 'scale' and scale_key:
        correction_function = partial(aclosest_pitch_from_scale, scale=scale_key)
    else:
        correction_function = closest_pitch

    pitch_corrected_y = autotune(
        y, sr, correction_function, plot,
        correction_strength, adaptive_strength, smooth_transitions
    )
    output_path = filepath.parent / (filepath.stem + '_pitch_corrected.wav')
    sf.write(str(output_path), pitch_corrected_y, sr)
    return pitch_corrected_y