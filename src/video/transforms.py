import cv2

def to_vertical_9_16(clip):
    """
    Converts a MoviePy clip into vertical 9:16 format (1080x1920).
    Video is resized by height and then center-cropped.
    """
    clip = clip.resize(height=1920)
    return clip.crop(width=1080, height=1920, x_center=clip.w // 2)

