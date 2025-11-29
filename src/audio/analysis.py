import librosa
import numpy as np

def detect_beats(audio_path):
    """
    Returns beat timestamps in seconds using librosa beat detection.
    """
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beats, sr=sr)

def get_audio_bands(audio_path, fps=30):
    """
    Extracts per-frame bass and hihat energy.
    """
    y, sr = librosa.load(audio_path)
    hop_length = int(sr / fps)

    # Short-time Fourier transform
    S = np.abs(librosa.stft(y))

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr)

    # Bass = 20–150 Hz
    bass_mask = (freqs >= 20) & (freqs <= 150)
    # Hihat = 5–10 kHz
    hihat_mask = (freqs >= 5000) & (freqs <= 10000)

    bass_energy = S[bass_mask].mean(axis=0)
    hihat_energy = S[hihat_mask].mean(axis=0)

    # Normalize
    bass_energy = bass_energy / (bass_energy.max() + 1e-6)
    hihat_energy = hihat_energy / (hihat_energy.max() + 1e-6)

    return bass_energy, hihat_energy

