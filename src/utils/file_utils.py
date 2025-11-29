import os
import random

def pick_random_file(folder, extensions):
    """
    Returns a random file from `folder` matching given extensions.
    """
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]
    if not files:
        raise FileNotFoundError(f"No files with {extensions} in {folder}")

    return os.path.join(folder, random.choice(files))

