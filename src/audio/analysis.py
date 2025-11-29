import librosa
import numpy as np

# --------------------------------------------
# Beat Detection
# --------------------------------------------

def detect_beats(audio_path):
    """
    Returns timestamps (in seconds) of detected beats.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return beat_times


# --------------------------------------------
# Bass & Hihat Energy Detection
# --------------------------------------------

def get_audio_bands(audio_path, fps=30):
    """
    Returns (bass_energy[], hihat_energy[]) per video frame.
    """

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    hop = int(sr / fps)

    # STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))

    freqs = librosa.fft_frequencies(sr=sr)

    bass_mask = (freqs >= 20) & (freqs <= 150)
    hihat_mask = (freqs >= 5000) & (freqs <= 12000)

    bass_energy = S[bass_mask].mean(axis=0)
    hihat_energy = S[hihat_mask].mean(axis=0)

    # Normalize
    if bass_energy.max() > 0:
        bass_energy = bass_energy / bass_energy.max()
    if hihat_energy.max() > 0:
        hihat_energy = hihat_energy / hihat_energy.max()

    # Replace NaN if silent
    bass_energy = np.nan_to_num(bass_energy)
    hihat_energy = np.nan_to_num(hihat_energy)

    return bass_energy, hihat_energy

