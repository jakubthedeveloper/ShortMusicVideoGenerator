import numpy as np
from pedalboard import Pedalboard, Reverb, Gain, Limiter
from pedalboard.io import AudioFile

def apply_audio_effects(audio_tuple):
    """
    Adds reverse reverb at start and reverb tail at end.

    audio_tuple = (audio, sr)
    audio = np.ndarray (samples x 2)
    """
    audio, sr = audio_tuple

    # --- Main reverb chain ---
    board = Pedalboard([
        Reverb(room_size=0.6, damping=0.2, wet_level=0.5, dry_level=0.8),
    ])

    processed = board(audio, sr)

    # --- Reverse reverb intro ---
    intro_len = sr // 2
    intro = processed[:intro_len][::-1] * 0.8

    # --- Reverb tail outro ---
    tail_len = sr // 2
    tail = processed[-tail_len:] * 0.8

    # --- Concatenate full audio ---
    final = np.concatenate([intro, processed, tail], axis=0)

    # --- FIRST normalization (pre-limiter) ---
    final = normalize_audio(final, peak=0.90)

    # --- Limiter to eliminate all clipping ---
    final = apply_final_limiter(final, sr)

    # --- FINAL normalization ---
    final = normalize_audio(final, peak=0.95)

    return final, sr


def normalize_audio(buffer, peak=0.95):
    """Normalize audio peak safely to avoid clipping."""
    max_val = np.max(np.abs(buffer))
    if max_val > 0:
        buffer = buffer * (peak / max_val)
    return buffer


def apply_final_limiter(buffer, sr):
    """Apply a safe soft limiter to prevent digital clipping."""
    board = Pedalboard([
        Limiter(threshold_db=-1.0, release_ms=50.0)
    ])
    return board(buffer, sr)

