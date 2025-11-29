import os
import random

def pick_random_file(folder, extensions):
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]
    if not files:
        raise FileNotFoundError(f"No files in {folder}")

    return os.path.join(folder, random.choice(files))

