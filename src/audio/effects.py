import numpy as np
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile

def apply_audio_effects(audio_tuple):
    """
    Adds reverse reverb at start and reverb tail at end.

    audio_tuple = (audio, sr)
    audio = np.ndarray (samples x 2)
    """
    audio, sr = audio_tuple

    board = Pedalboard([
        Reverb(room_size=0.6, damping=0.2, wet_level=0.5, dry_level=0.8),
    ])

    processed = board(audio, sr)

    # Reverse reverb intro
    intro_len = sr // 2
    intro = processed[:intro_len][::-1] * 0.8

    # Reverb tail outro
    tail_len = sr // 2
    tail = processed[-tail_len:] * 0.8

    final = np.concatenate([intro, processed, tail], axis=0)

    return final, sr

