import moviepy as mp
import numpy as np

def to_vertical_9_16(clip):
    """Convert any clip to 1080x1920 (9:16) vertical format."""

    target_w = 1080
    target_h = 1920
    target_ratio = target_w / target_h

    w, h = clip.size
    src_ratio = w / h

    # Resize depending on aspect ratio
    if src_ratio > target_ratio:
        # Source is wider — scale by height
        clip = clip.resized(height=target_h)
        new_w, _ = clip.size
        x = int((new_w - target_w) / 2)
        clip = clip.cropped(x1=x, x2=x + target_w)

    else:
        # Source is taller — scale by width
        clip = clip.resized(width=target_w)
        _, new_h = clip.size
        y = int((new_h - target_h) / 2)
        clip = clip.cropped(y1=y, y2=y + target_h)

    return clip

