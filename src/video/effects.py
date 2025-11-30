import numpy as np
import cupy as cp
import cv2

# ---- GPU helpers ----

def np_to_cp(arr):
    """Convert NumPy -> CuPy."""
    return cp.asarray(arr)


def cp_to_np(arr):
    """Convert CuPy -> NumPy."""
    return cp.asnumpy(arr)


def to_float_gpu(frame_gpu):
    return frame_gpu.astype(cp.float32) / 255.0


def to_uint_gpu(frame_gpu):
    return cp.clip(frame_gpu * 255, 0, 255).astype(cp.uint8)


def ensure_rgb(arr):
    """
    Ensures output is always (H, W, 3).
    Fixes GPU kaleidoscope outputs that may return 1-channel or 4-channel results.
    """
    if arr.ndim == 2:
        # grayscale -> RGB
        return np.stack([arr, arr, arr], axis=-1)

    if arr.shape[2] == 1:
        # (H,W,1) -> (H,W,3)
        return np.repeat(arr, 3, axis=2)

    if arr.shape[2] == 4:
        # (H,W,4) -> discard alpha / extra channel
        return arr[..., :3]

    return arr


# ======================================================================
# ========================= GPU VIDEO EFFECTS ===========================
# ======================================================================


# ----------------------------------------------------------------------
# 1. HUE SHIFT (GPU)
# ----------------------------------------------------------------------
def hue_shift(frame, t, beats, bass, hihat):
    # Convert to HSV (CPU), but modify GPU-side for speed
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_gpu = np_to_cp(hsv)

    hsv_gpu[..., 0] = (hsv_gpu[..., 0] + t * 8) % 180

    hsv_cpu = cp_to_np(hsv_gpu)
    return cv2.cvtColor(hsv_cpu.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ----------------------------------------------------------------------
# 2. NEON PULSE (GPU)
# ----------------------------------------------------------------------
def neon_pulse(frame, t, beats, bass, hihat):
    f = np_to_cp(frame).astype(cp.float32) / 255.0

    pulse = 0.5 + 0.5 * cp.sin(t * 6 + bass * 5)
    f = cp.power(f * 1.6, 2.0) * pulse

    return cp_to_np(to_uint_gpu(f))


# ----------------------------------------------------------------------
# 3. SOLARIZE (GPU)
# ----------------------------------------------------------------------
def solarize(frame, t, beats, bass, hihat):
    f = np_to_cp(frame)
    threshold = int(128 + bass * 80)

    mask = f > threshold
    f[mask] = 255 - f[mask]

    return cp_to_np(f)


# ----------------------------------------------------------------------
# 4. RGB SPLIT (GPU)
# ----------------------------------------------------------------------
def rgb_split(frame, t, beats, bass, hihat):
    f = np_to_cp(frame)
    shift = int(5 + bass * 10)

    b = f[..., 0]
    g = cp.roll(f[..., 1], shift, axis=1)
    r = cp.roll(f[..., 2], -shift, axis=1)

    out = cp.stack([b, g, r], axis=-1)
    return cp_to_np(out)


# ----------------------------------------------------------------------
# 5. BLOCK GLITCH (hybrid GPU/CPU)
# ----------------------------------------------------------------------
def block_glitch(frame, t, beats, bass, hihat):
    f = frame.copy()
    h, w = frame.shape[:2]
    block = 40
    amount = int(3 + bass * 8)

    for _ in range(amount):
        y = np.random.randint(0, h - block)
        x = np.random.randint(0, w - block)
        dx = np.random.randint(-20, 20)
        dy = np.random.randint(-20, 20)

        y2 = np.clip(y+dy, 0, h-block)
        x2 = np.clip(x+dx, 0, w-block)
        f[y:y+block, x:x+block] = f[y2:y2+block, x2:x2+block]

    return f


# ----------------------------------------------------------------------
# 6. SCANLINES (GPU)
# ----------------------------------------------------------------------
def scanlines(frame, t, beats, bass, hihat):
    f = np_to_cp(frame)
    h = frame.shape[0]

    mask = cp.zeros_like(f)
    mask[::2] = 40

    out = cp.clip(f - mask, 0, 255)
    return cp_to_np(out.astype(cp.uint8))


# ----------------------------------------------------------------------
# 7. SWIRL (GPU + CPU remap)
# ----------------------------------------------------------------------
def swirl(frame, t, beats, bass, hihat):
    f_gpu = np_to_cp(frame)
    h, w = frame.shape[:2]

    y, x = cp.indices((h, w))
    cx, cy = w // 2, h // 2

    x = x - cx
    y = y - cy

    radius = cp.sqrt(x*x + y*y)
    angle = cp.arctan2(y, x) + (radius / 300) * 0.6

    map_x = (cx + radius * cp.cos(angle)).astype(cp.float32)
    map_y = (cy + radius * cp.sin(angle)).astype(cp.float32)

    # CPU remap
    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(map_x),
        cp_to_np(map_y),
        cv2.INTER_LINEAR
    )


# ----------------------------------------------------------------------
# 8. WAVE (GPU + CPU remap)
# ----------------------------------------------------------------------
def wave(frame, t, beats, bass, hihat):
    f_gpu = np_to_cp(frame)
    h, w = frame.shape[:2]

    y, x = cp.indices((h, w))
    wave_x = x + cp.sin(y / 20 + t * 5) * 10
    wave_y = y + cp.sin(x / 30 + t * 4) * 10

    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(wave_x.astype(cp.float32)),
        cp_to_np(wave_y.astype(cp.float32)),
        cv2.INTER_LINEAR
    )


# ----------------------------------------------------------------------
# 9. RIPPLE (GPU + CPU remap)
# ----------------------------------------------------------------------
def ripple(frame, t, beats, bass, hihat):
    f_gpu = np_to_cp(frame)
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    y, x = cp.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = cp.sqrt(dx*dx + dy*dy)

    ripple = cp.sin(r/6 - t*10) * 5

    map_x = (x + ripple).astype(cp.float32)
    map_y = (y + ripple).astype(cp.float32)

    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(map_x),
        cp_to_np(map_y),
        cv2.INTER_LINEAR
    )


# ----------------------------------------------------------------------
# 10. KALEIDOSCOPE (GPU + CPU remap)
# ----------------------------------------------------------------------
def kaleidoscope_mirrored(frame, t, beats, bass, hihat, segments=6):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(cx, cy)

    output = np.zeros_like(frame)
    step = 360 / segments

    yy, xx = np.indices((h, w))
    x = xx - cx
    y = yy - cy

    angle = (np.degrees(np.arctan2(y, x)) + 360) % 360
    dist = np.sqrt(x*x + y*y)
    mask = dist < radius

    for i in range(segments):
        start_angle = i * step
        end_angle = start_angle + step

        wedge_mask = (angle >= start_angle) & (angle < end_angle) & mask
        if not np.any(wedge_mask):
            continue

        M = cv2.getRotationMatrix2D((cx, cy), -start_angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))

        if i % 2 == 1:
            rotated = cv2.flip(rotated, 1)

        output[wedge_mask] = rotated[wedge_mask]

    return output

def kaleidoscope_dynamic(frame, t, beats, bass, hihat):
    # dynamic segment count 5 ↔ 14
    segments = int(5 + bass * 6 + hihat * 3)
    segments = max(3, min(segments, 18))
    return kaleidoscope_mirrored(frame, t, beats, bass, hihat, segments)

def kaleidoscope_liquid(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Distortion strength reacts to bass
    strength = 0.005 + bass * 0.02
    
    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = yy - cy

    # radial wave
    r = np.sqrt(dx*dx + dy*dy)
    wave = np.sin(r * 0.03 + t * 3) * strength * r

    map_x = (xx + dx * wave).astype(np.float32)
    map_y = (yy + dy * wave).astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    return kaleidoscope_mirrored(warped, t, beats, bass, hihat, segments=8)

def kaleidoscope_3d(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # curvature reacts to bass
    k = 0.00015 + bass * 0.0004

    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = yy - cy
    r2 = dx*dx + dy*dy

    # distortion factor
    factor = 1 + k * r2
    map_x = (dx / factor + cx).astype(np.float32)
    map_y = (dy / factor + cy).astype(np.float32)

    curved = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    return kaleidoscope_mirrored(curved, t, beats, bass, hihat, segments=10)

def kaleidoscope_fractal(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # fractal zoom speed
    zoom = 1 + 0.03 * np.sin(t * 1.5 + bass * 5)

    M = cv2.getRotationMatrix2D((cx, cy), t * 20, zoom)
    rotated = cv2.warpAffine(frame, M, (w, h))

    # fractal layering
    blend = cv2.addWeighted(frame, 0.5, rotated, 0.5, 0)
    return kaleidoscope_mirrored(blend, t, beats, bass, hihat, segments=12)


# ======================================================================
# KALEIDOSCOPES – GPU versions (PyTorch CUDA)
# ======================================================================

# Import CUDA kaleidoscopes
try:
    from video.gpu.kaleidoscope_cuda import (
        kaleidoscope_mirrored_cuda,
        kaleidoscope_dynamic_cuda,
        kaleidoscope_liquid_cuda,
        kaleidoscope_3d_cuda,
        kaleidoscope_fractal_cuda,
    )
    USE_KALEIDOSCOPE_CUDA = True
except Exception as e:
    print("[WARNING] CUDA kaleidoscopes unavailable:", e)
    USE_KALEIDOSCOPE_CUDA = False


# Wrappers: Auto-select CUDA or CPU
def kaleidoscope_mirrored_auto(frame, t, beats, bass, hihat):
    if USE_KALEIDOSCOPE_CUDA:
        return kaleidoscope_mirrored_cuda(frame, t, beats, bass, hihat)
    return kaleidoscope_mirrored(frame, t, beats, bass, hihat)


def kaleidoscope_dynamic_auto(frame, t, beats, bass, hihat):
    if USE_KALEIDOSCOPE_CUDA:
        return kaleidoscope_dynamic_cuda(frame, t, beats, bass, hihat)
    return kaleidoscope_dynamic(frame, t, beats, bass, hihat)


def kaleidoscope_liquid_auto(frame, t, beats, bass, hihat):
    if USE_KALEIDOSCOPE_CUDA:
        return kaleidoscope_liquid_cuda(frame, t, beats, bass, hihat)
    return kaleidoscope_liquid(frame, t, beats, bass, hihat)


def kaleidoscope_3d_auto(frame, t, beats, bass, hihat):
    if USE_KALEIDOSCOPE_CUDA:
        return kaleidoscope_3d_cuda(frame, t, beats, bass, hihat)
    return kaleidoscope_3d(frame, t, beats, bass, hihat)


def kaleidoscope_fractal_auto(frame, t, beats, bass, hihat):
    """
    Safe wrapper for CUDA or CPU fractal kaleidoscope.
    Ensures correct channel dimensions even if CUDA kernel returns 1 or 4 channels.
    """
    if USE_KALEIDOSCOPE_CUDA:
        try:
            out = kaleidoscope_fractal_cuda(frame, t, beats, bass, hihat)
            return ensure_rgb(out)     # <-- FIXED
        except Exception as e:
            print("[WARNING] CUDA fractal failed, falling back to CPU:", e)

    out = kaleidoscope_fractal(frame, t, beats, bass, hihat)
    return ensure_rgb(out)

# ----------------------------------------------------------------------
# 11. MANDALA TWIST (GPU + CPU remap)
# ----------------------------------------------------------------------
def mandala_twist(frame, t, beats, bass, hihat):
    f_gpu = np_to_cp(frame)
    h, w = frame.shape[:2]

    y, x = cp.indices((h, w))
    cx, cy = w // 2, h // 2

    dx = x - cx
    dy = y - cy
    radius = cp.sqrt(dx*dx + dy*dy)

    angle = cp.arctan2(dy, dx) + cp.sin(t*3 + radius/50) * 0.5

    map_x = (cx + radius * cp.cos(angle)).astype(cp.float32)
    map_y = (cy + radius * cp.sin(angle)).astype(cp.float32)

    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(map_x),
        cp_to_np(map_y),
        cv2.INTER_LINEAR
    )


# ----------------------------------------------------------------------
# 12. PIXEL SORT (GPU)
# ----------------------------------------------------------------------
def pixel_sort(frame, t, beats, bass, hihat):
    f = np_to_cp(frame)
    h, w = frame.shape[:2]

    out = f.copy()

    for y in range(h):
        row = out[y]
        lum = row.mean(axis=1)
        idx = cp.argsort(lum)
        out[y] = row[idx]

    return cp_to_np(out)


# ----------------------------------------------------------------------
# 13. PIXEL SORT VERTICAL (GPU)
# ----------------------------------------------------------------------
def pixel_sort_vertical(frame, t, beats, bass, hihat):
    f = np_to_cp(frame)
    h, w = frame.shape[:2]

    out = f.copy()

    for x in range(w):
        col = out[:, x]
        lum = col.mean(axis=1)
        idx = cp.argsort(lum)
        out[:, x] = col[idx]

    return cp_to_np(out)


# ----------------------------------------------------------------------
# 14. BEAT GLOW (GPU)
# ----------------------------------------------------------------------
def beat_glow(frame, t, beats, bass, hihat):
    f = np_to_cp(frame).astype(cp.float32) / 255.0
    power = 1.0 + bass * 1.5
    out = cp.power(f, power)
    return cp_to_np(to_uint_gpu(out))


# ----------------------------------------------------------------------
# 15. BEAT RGB SHAKE (GPU)
# ----------------------------------------------------------------------
def beat_rgb_shake(frame, t, beats, bass, hihat):
    shift = int(hihat * 10)

    f = np_to_cp(frame)

    b = cp.roll(f[..., 0], shift, axis=0)
    g = cp.roll(f[..., 1], -shift, axis=1)
    r = f[..., 2]

    return cp_to_np(cp.stack([b, g, r], axis=-1))


# ----------------------------------------------------------------------
# 16. BEAT ZOOM (GPU + CPU remap)
# ----------------------------------------------------------------------
def beat_zoom(frame, t, beats, bass, hihat):
    scale = 1.0 + bass * 0.1

    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    x = (new_w - w) // 2
    y = (new_h - h) // 2

    cropped = resized[y:y+h, x:x+w]
    return cropped


# ----------------------------------------------------------------------
# 17. BEAT KALEIDO PULSE (GPU + CPU remap)
# ----------------------------------------------------------------------
def beat_kaleido_pulse(frame, t, beats, bass, hihat):
    intensity = 1 + bass * 0.6
    f_gpu = np_to_cp(frame)

    h, w = frame.shape[:2]
    y, x = cp.indices((h, w))
    cx, cy = w//2, h//2

    dx = x - cx
    dy = y - cy
    radius = cp.sqrt(dx*dx + dy*dy)

    angle = cp.arctan2(dy, dx)
    angle = (angle * intensity) % (cp.pi/4)

    map_x = (cx + radius * cp.cos(angle)).astype(cp.float32)
    map_y = (cy + radius * cp.sin(angle)).astype(cp.float32)

    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(map_x),
        cp_to_np(map_y),
        cv2.INTER_LINEAR
    )


# ----------------------------------------------------------------------
# 18. BEAT RIPPLE (GPU + CPU remap)
# ----------------------------------------------------------------------
def beat_ripple(frame, t, beats, bass, hihat):
    f_gpu = np_to_cp(frame)
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = cp.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = cp.sqrt(dx*dx + dy*dy)

    ripple = cp.sin(r/10 - t*20) * bass * 20

    map_x = (x + ripple).astype(cp.float32)
    map_y = (y + ripple).astype(cp.float32)

    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(map_x),
        cp_to_np(map_y),
        cv2.INTER_LINEAR
    )


# ----------------------------------------------------------------------
# 19. BASS ZOOM (CPU)
# ----------------------------------------------------------------------
def bass_zoom(frame, t, beats, bass, hihat):
    scale = 1.0 + bass * 0.3

    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    x = (new_w - w) // 2
    y = (new_h - h) // 2

    return resized[y:y+h, x:x+w]


# ----------------------------------------------------------------------
# 20. BASS DISTORT (GPU)
# ----------------------------------------------------------------------
def bass_distort(frame, t, beats, bass, hihat):
    f = np_to_cp(frame).astype(cp.float32) / 255.0

    noise = cp.random.uniform(-1, 1, frame.shape, dtype=cp.float32)
    distorted = cp.clip(f + noise * (bass * 0.2), 0, 1)

    return cp_to_np(to_uint_gpu(distorted))


# ======================================================================
# EFFECT LIST
# ======================================================================

VIDEO_EFFECTS = [
    # fully CUDA-accelerated kaleidoscopes
    kaleidoscope_mirrored_auto,
    kaleidoscope_dynamic_auto,
    kaleidoscope_liquid_auto,
    kaleidoscope_3d_auto,
    kaleidoscope_fractal_auto,

    # your GPU / CuPy effects
    hue_shift,
    neon_pulse,
    solarize,
    rgb_split,
    block_glitch,
    scanlines,
    swirl,
    wave,
    ripple,
    mandala_twist,
    pixel_sort,
    pixel_sort_vertical,
    beat_glow,
    beat_rgb_shake,
    beat_zoom,
    beat_kaleido_pulse,
    beat_ripple,
    bass_zoom,
    bass_distort,
]

