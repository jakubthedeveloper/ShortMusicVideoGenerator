import numpy as np
from pedalboard import Pedalboard, Reverb

# --------------------------------------------
# Reverse Reverb Intro
# --------------------------------------------

def reverse_reverb_intro(y, sr, intro_ms=400):
    """
    Creates a cinematic reverse-reverb swell at the beginning of the audio.
    """
    intro_samples = int(sr * intro_ms / 1000)

    if intro_samples > len(y):
        intro_samples = len(y) // 2

    segment = y[:intro_samples]
    reversed_seg = segment[::-1]

    board = Pedalboard([Reverb(room_size=0.6)])
    processed = board(reversed_seg, sr)
    processed = processed[::-1]

    fade = np.linspace(0, 1, intro_samples).reshape(-1, 1)
    padded = processed * fade

    return np.vstack([padded, y])


# --------------------------------------------
# Reverb Tail Outro
# --------------------------------------------

def reverb_tail(y, sr, tail_ms=600):
    """
    Adds natural reverb decay at the end of the audio.
    """
    tail_samples = int(sr * tail_ms / 1000)

    if tail_samples > len(y):
        tail_samples = len(y) // 2

    segment = y[-tail_samples:]
    board = Pedalboard([Reverb(room_size=0.5)])
    processed = board(segment, sr)

    fade = np.linspace(1, 0, tail_samples).reshape(-1, 1)
    processed = processed * fade

    return np.vstack([y, processed])


# --------------------------------------------
# Apply all audio effects
# --------------------------------------------

def apply_audio_effects(data):
    """
    Applies all audio processing:
    1) Reverse reverb intro
    2) Reverb tail outro
    """

    y, sr = data

    y = reverse_reverb_intro(y, sr)
    y = reverb_tail(y, sr)

    return y, sr

