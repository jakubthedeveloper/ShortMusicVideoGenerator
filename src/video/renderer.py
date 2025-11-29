import cv2

def apply_video_effect(frame, t, beats, bass, hihat, effect_fn):
    """
    Applies an effect function to a single frame.

    MoviePy 2.x frames are delivered as RGB arrays.
    OpenCV expects BGR, so we convert back and forth.

    Parameters:
        frame (np.ndarray): RGB frame from MoviePy
        t (float): current timestamp in seconds
        beats (list): beat timestamps detected from audio
        bass (float): bass band energy
        hihat (float): hihat band energy
        effect_fn (callable): selected effect function

    Returns:
        np.ndarray: RGB frame after processing
    """
    # Convert RGB → BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Apply effect
    processed_bgr = effect_fn(frame_bgr, t, beats, bass, hihat)

    # Convert back BGR → RGB
    return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

