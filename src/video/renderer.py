import cv2

def apply_video_effect(frame, t, beats, bass, hihat, effect_fn):
    """
    Applies a selected effect function to a single RGB frame.
    MoviePy frames are RGB → convert to BGR for OpenCV.
    """
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    processed_bgr = effect_fn(frame_bgr, t, beats, bass, hihat)
    return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

