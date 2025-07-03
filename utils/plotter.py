import matplotlib.pyplot as plt
import numpy as np

def create_plot(time: 'np.ndarray', original_f0: 'np.ndarray', target_f0: 'np.ndarray', output_path="pitch_correction.png"):
    """
    Generates and saves a plot comparing original and target pitch.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_f0, label='Original Detected Pitch (CREPE)', color='cyan', alpha=0.7, linewidth=1.5)
    plt.plot(time, target_f0, label='New Target Pitch Contour', color='orange', linewidth=2)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("Pitch Correction Analysis")
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()