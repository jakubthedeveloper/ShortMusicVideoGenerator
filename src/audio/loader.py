import soundfile as sf
import numpy as np

def load_audio(path):
    """
    Load audio from a file using soundfile.
    Output shape is always (N, 2) for stereo compatibility.
    """
    y, sr = sf.read(path, always_2d=True)

    # Normalize audio to float32
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    return y, sr

