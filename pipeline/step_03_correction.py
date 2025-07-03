import numpy as np
import librosa
from scipy.signal import medfilt

SEMITONES_IN_OCTAVE = 12

def create_target_pitch_contour(f0: 'np.ndarray', time: 'np.ndarray', sr: int, scale_key: str, glide_ms: int) -> 'np.ndarray':
    """
    Generates a new, musically-sensible pitch contour that preserves natural vibrato and micro-intonation.
    Only corrects when deviation is significant and sustained (>25 cents for >20ms).
    """
    if np.all(np.isnan(f0)):
        return f0

    original_midi = librosa.hz_to_midi(f0)
    
    # Treat any detected frequency below 1 Hz (including 0) as invalid (NaN)
    original_midi[f0 < 1] = np.nan
    
    target_midi = np.copy(original_midi)

    # --- INTELLIGENT CORRECTION PLANNING ---
    # Only correct when deviation is significant and sustained
    hop_time = time[1] - time[0]  # Time per frame
    min_duration_frames = int(0.020 / hop_time)  # 20ms minimum
    threshold_cents = 25  # Only correct if off by more than 25 cents
    
    # Calculate deviation from nearest note
    if scale_key == "closest":
        possible_notes = np.arange(0, 128) 
    else:
        degrees = librosa.key_to_degrees(scale_key)
        possible_notes = []
        for octave in range(10):
            for degree in degrees:
                possible_notes.append(12 * octave + degree)
        possible_notes = np.array(possible_notes)

    # Find nearest notes and calculate deviations
    deviations = np.full(len(original_midi), np.inf)
    nearest_notes = np.full(len(original_midi), np.nan)
    
    for i in range(len(original_midi)):
        if np.isnan(original_midi[i]):
            continue
        
        closest_note_index = np.argmin(np.abs(possible_notes - original_midi[i]))
        nearest_note = possible_notes[closest_note_index]
        nearest_notes[i] = nearest_note
        
        # Calculate deviation in cents
        deviation_cents = (original_midi[i] - nearest_note) * 100
        deviations[i] = abs(deviation_cents)
    
    # Apply median filter to smooth deviations (preserve vibrato)
    deviations_smooth = medfilt(deviations, kernel_size=5)
    
    # Create correction mask: only correct if deviation is large and sustained
    correction_mask = np.zeros(len(original_midi), dtype=bool)
    
    for i in range(len(original_midi)):
        if np.isnan(original_midi[i]) or np.isnan(nearest_notes[i]):
            continue
            
        # Check if deviation is above threshold
        if deviations_smooth[i] > threshold_cents:
            # Check if it's sustained for minimum duration
            start_idx = max(0, i - min_duration_frames // 2)
            end_idx = min(len(original_midi), i + min_duration_frames // 2)
            
            # If most frames in this window are above threshold, mark for correction
            window_deviations = deviations_smooth[start_idx:end_idx]
            if np.mean(window_deviations > threshold_cents) > 0.7:  # 70% of frames
                correction_mask[i] = True
    
    # Apply corrections only where mask is True
    for i in range(len(original_midi)):
        if correction_mask[i] and not np.isnan(nearest_notes[i]):
            target_midi[i] = nearest_notes[i]
        else:
            # Keep original pitch (preserve vibrato and micro-intonation)
            target_midi[i] = original_midi[i]
        
    # --- SMOOTH TRANSITIONS ---
    # Apply attack/release envelopes for smooth note transitions
    hop_length_s = time[1] - time[0]
    attack_frames = int((15 / 1000.0) / hop_length_s)  # 15ms attack
    release_frames = int((80 / 1000.0) / hop_length_s)  # 80ms release
    
    # Find note change points
    note_changes = np.where(np.abs(np.diff(np.nan_to_num(target_midi))) > 0)[0]
    
    for change_point in note_changes:
        # Apply attack envelope
        start_frame = max(0, change_point - attack_frames)
        end_frame = min(len(target_midi), change_point + attack_frames)
        
        if start_frame < end_frame:
            # Create smooth transition
            transition = np.linspace(0, 1, end_frame - start_frame)
            original_pitch = target_midi[start_frame]
            target_pitch = target_midi[end_frame-1]
            
            if not (np.isnan(original_pitch) or np.isnan(target_pitch)):
                target_midi[start_frame:end_frame] = (
                    original_pitch * (1 - transition) + target_pitch * transition
                )

    target_f0 = librosa.midi_to_hz(target_midi)
    
    return target_f0