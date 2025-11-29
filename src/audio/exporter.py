import soundfile as sf
import tempfile

def save_temp_audio(audio_tuple):
    """
    Saves processed audio to a temp WAV file.
    Returns path to file.
    """
    audio, sr = audio_tuple

    fd, path = tempfile.mkstemp(suffix=".wav", dir="temp")
    sf.write(path, audio, sr)
    return path

