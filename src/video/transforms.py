def to_vertical_9_16(clip):
    """
    Converts a MoviePy clip to 1080x1920 (vertical 9:16).

    Strategy:
    - resize by height = 1920
    - center-crop width to 1080
    """
    # Resize so height = 1920
    clip = clip.resize(height=1920)

    # Center crop width
    return clip.crop(
        width=1080,
        height=1920,
        x_center=clip.w / 2,
        y_center=clip.h / 2
    )

