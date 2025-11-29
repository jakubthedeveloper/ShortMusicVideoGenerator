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

    hsv_gpu[..., 0] = (hsv_gpu[..., 0] + t * 30) % 180

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
    angle = cp.arctan2(y, x) + (radius / 300) * 2

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
def kaleidoscope(frame, t, beats, bass, hihat):
    f_gpu = np_to_cp(frame)
    h, w = frame.shape[:2]

    y, x = cp.indices((h, w))
    cx, cy = w // 2, h // 2

    dx = x - cx
    dy = y - cy
    angle = cp.arctan2(dy, dx)
    radius = cp.sqrt(dx*dx + dy*dy)

    angle = (angle + cp.pi) % (cp.pi/3)
    angle = angle - cp.pi

    map_x = (cx + radius * cp.cos(angle)).astype(cp.float32)
    map_y = (cy + radius * cp.sin(angle)).astype(cp.float32)

    return cv2.remap(
        cp_to_np(f_gpu),
        cp_to_np(map_x),
        cp_to_np(map_y),
        cv2.INTER_LINEAR
    )


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
    intensity = 1 + bass * 2
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
    hue_shift,
    neon_pulse,
    solarize,
    rgb_split,
    block_glitch,
    scanlines,
    swirl,
    wave,
    ripple,
    kaleidoscope,
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

