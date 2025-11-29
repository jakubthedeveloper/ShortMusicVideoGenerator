import soundfile as sf
import numpy as np

def load_audio(path):
    """
    Loads audio into a numpy array for DSP processing.
    Returns stereo float32 array and sample rate.

    For MoviePy 2.x, audio has to be processed using an external
    library because MoviePy removed many internal audio loaders.
    """
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32)
    return audio, sr

