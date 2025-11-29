import os
import re
from mutagen import File as MutagenFile

def clean_filename(name: str) -> str:
    """Convert a filename like '03. My Song (final master).mp3' into 'My Song'."""
    
    # Remove extension
    name = os.path.splitext(name)[0]

    # Remove leading track numbers ("01", "03.", "12 -")
    name = re.sub(r"^\d+\s*[-_.]?\s*", "", name)

    # Remove unwanted suffixes
    remove_suffixes = [
        "final", "master", "mix", "version", "edit",
        "remaster", "remix", "draft", "v2", "v3", "copy"
    ]
    for s in remove_suffixes:
        name = re.sub(rf"\b{s}\b", "", name, flags=re.IGNORECASE)

    # Remove parentheses if empty/leftover
    name = re.sub(r"\(\s*\)", "", name)

    # Replace multiple spaces with single
    name = re.sub(r"\s{2,}", " ", name)

    return name.strip()


def read_audio_metadata_title(path: str) -> str | None:
    """
    Reads title from audio metadata using Mutagen.
    Returns None if no title found.
    """
    try:
        audio = MutagenFile(path)
        if not audio:
            return None

        # ID3-based formats
        if hasattr(audio, "tags") and audio.tags:
            # Common ID3 frames
            for key in ("TIT2", "TITLE", "\xa9nam"):
                if key in audio.tags:
                    value = audio.tags[key]
                    if isinstance(value, list):
                        return value[0].strip()
                    return str(value).strip()

        return None
    except Exception:
        return None


def generate_title_from_audio(path: str) -> str:
    """
    Returns the final title to be used in the Short:
    1. Tries reading metadata Title
    2. Falls back to filename cleaning
    """
    title = read_audio_metadata_title(path)
    if title:
        return title
    
    # fallback
    filename = os.path.basename(path)
    return clean_filename(filename)
