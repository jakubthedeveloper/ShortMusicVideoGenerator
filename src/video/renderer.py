import cv2
import numpy as np

def apply_video_effect(frame, t, beats, bass, hihat, effect_fn):
    """
    Applies an effect or effect chain to a single frame.

    Supports two modes:
      • single effect function
      • list of effect functions (effect chain)

    Parameters:
        frame (np.ndarray): RGB frame from MoviePy
        t (float): current timestamp in seconds
        beats (list): timestamps of detected beats
        bass (float): bass band energy
        hihat (float): hihat band energy
        effect_fn: callable OR list[callable]

    Returns:
        np.ndarray: RGB frame after processing
    """

    # Convert RGB → BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # If we got a single function → wrap it into a list
    if callable(effect_fn):
        chain = [effect_fn]
    else:
        chain = effect_fn  # already a list

    # Apply chain
    for fn in chain:
        frame_bgr = fn(frame_bgr, t, beats, bass, hihat)
        if frame_bgr is None:
            raise ValueError(f"Effect {fn.__name__} returned None!")

    # Convert BGR → RGB back for MoviePy
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

