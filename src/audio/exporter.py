import soundfile as sf
import os

def save_temp_audio(data):
    """
    Saves processed audio to temp/temp_audio.wav and returns the path.
    """
    y, sr = data

    os.makedirs("temp", exist_ok=True)
    path = "temp/temp_audio.wav"

    sf.write(path, y, sr)

    return path

