import cv2
import numpy as np
import random

# ---------------------------------------
# Helpers
# ---------------------------------------

def to_float(frame):
    return frame.astype(np.float32) / 255.0

def to_uint(frame):
    return np.clip(frame * 255, 0, 255).astype(np.uint8)


# =======================================================
# 🎨 COLOR EFFECTS
# =======================================================

def hue_shift(frame, t, beats, bass, hihat):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + t * 30) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def neon_pulse(frame, t, beats, bass, hihat):
    f = to_float(frame)
    pulse = 0.5 + 0.5 * np.sin(t * 6 + bass * 5)
    f = np.power(f * 1.6, 2.0) * pulse
    return to_uint(f)


def solarize(frame, t, beats, bass, hihat):
    threshold = int(128 + 40 * np.sin(t * 4))
    mask = frame > threshold
    out = frame.copy()
    out[mask] = 255 - out[mask]
    return out


# =======================================================
# ⚡ GLITCH EFFECTS
# =======================================================

def rgb_split(frame, t, beats, bass, hihat):
    b, g, r = cv2.split(frame)
    shift = int(8 * np.sin(t * 5) + bass * 12)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=0)
    return cv2.merge([b, g, r])


def block_glitch(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    out = frame.copy()
    for _ in range(10 + int(10 * hihat)):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        bw = random.randint(20, 80)
        bh = random.randint(10, 40)
        dx = random.randint(-15, 15)
        block = frame[y:y+bh, x:x+bw]
        out[y:y+bh, x+dx:x+dx+bw] = block
    return out


def scanlines(frame, t, beats, bass, hihat):
    out = frame.copy()
    for y in range(0, frame.shape[0], 3):
        out[y] = (out[y] * 0.35).astype(np.uint8)
    return out


# =======================================================
# 🌀 DISTORTION EFFECTS
# =======================================================

def swirl(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy

    radius = np.sqrt(x*x + y*y)
    angle = np.arctan2(y, x) + (radius / 300) * 2

    map_x = (cx + radius * np.cos(angle)).astype(np.float32)
    map_y = (cy + radius * np.sin(angle)).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def wave(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Y2 = Y + 15 * np.sin(2*np.pi*(X/150 + t))
    return frame[Y2.clip(0, h-1).astype(int), X]


def ripple(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx*dx + dy*dy)

    ripple = np.sin(r/6 - t*10) * 5

    map_x = (x + ripple).astype(np.float32)
    map_y = (y + ripple).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


# =======================================================
# ✴️ KALEIDOSCOPES & MANDALAS
# =======================================================

def kaleidoscope(frame, t, beats, bass, hihat, segments=6):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy

    angle = np.arctan2(dy, dx)
    radius = np.sqrt(dx*dx + dy*dy)

    angle = (angle % (2*np.pi / segments)) * segments

    map_x = (cx + radius * np.cos(angle)).astype(np.float32)
    map_y = (cy + radius * np.sin(angle)).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def mandala_twist(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx*dx + dy*dy)

    angle = np.arctan2(dy, dx) + np.sin(r/50 + t*5) * 2

    map_x = (cx + r * np.cos(angle)).astype(np.float32)
    map_y = (cy + r * np.sin(angle)).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


# =======================================================
# 📊 PIXEL SORTING
# =======================================================

def pixel_sort(frame, t, beats, bass, hihat):
    out = frame.copy()
    for y in range(frame.shape[0]):
        row = frame[y]
        lum = row.mean(axis=1)
        idx = np.argsort(lum)
        out[y] = row[idx]
    return out


def pixel_sort_vertical(frame, t, beats, bass, hihat):
    f = frame.transpose((1,0,2))
    out = pixel_sort(f, t, beats, bass, hihat)
    return out.transpose((1,0,2))


# =======================================================
# 🎧 BEAT-REACTIVE EFFECTS
# =======================================================

def beat_glow(frame, t, beats, bass, hihat):
    if len(beats) == 0:
        dist = 1
    else:
        dist = np.min(np.abs(beats - t))

    pulse = np.exp(-dist * 15)  # sharp beat glow
    f = frame.astype(np.float32)
    f *= (1.0 + pulse * 2.2)
    return np.clip(f, 0, 255).astype(np.uint8)


def beat_rgb_shake(frame, t, beats, bass, hihat):
    if len(beats) == 0:
        dist = 1
    else:
        dist = np.min(np.abs(beats - t))

    amount = max(0, 12 - dist * 120)
    b, g, r = cv2.split(frame)
    r = np.roll(r, int(amount), axis=1)
    b = np.roll(b, -int(amount), axis=0)
    return cv2.merge([b, g, r])


def beat_zoom(frame, t, beats, bass, hihat):
    if len(beats) == 0:
        return frame

    dist = np.min(np.abs(beats - t))
    strength = max(0, 1.2 - dist * 10)

    if strength <= 0.01:
        return frame

    h, w = frame.shape[:2]
    scale = 1 + strength * 0.3

    nh, nw = int(h / scale), int(w / scale)
    y1, x1 = (h - nh)//2, (w - nw)//2
    cropped = frame[y1:y1+nh, x1:x1+nw]
    return cv2.resize(cropped, (w, h))


def beat_kaleido_pulse(frame, t, beats, bass, hihat):
    if len(beats) == 0:
        return kaleidoscope(frame, t, beats, bass, hihat, 6)

    dist = np.min(np.abs(beats - t))
    seg = 6 + int(np.exp(-dist * 20) * 10)
    return kaleidoscope(frame, t, beats, bass, hihat, seg)


def beat_ripple(frame, t, beats, bass, hihat):
    if len(beats) == 0:
        dist = 1
    else:
        dist = np.min(np.abs(beats - t))

    amp = 20 * np.exp(-dist * 15)

    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx*dx + dy*dy)

    ripple = np.sin(r/8 - t*10) * amp

    map_x = (x + ripple).astype(np.float32)
    map_y = (y + ripple).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


# =======================================================
# 🎚 BASS-REACTIVE EFFECTS
# =======================================================

def bass_zoom(frame, t, beats, bass, hihat):
    strength = bass * 0.5
    if strength < 0.01:
        return frame

    h, w = frame.shape[:2]
    scale = 1 + strength

    nh, nw = int(h / scale), int(w / scale)
    y1, x1 = (h - nh)//2, (w - nw)//2
    crop = frame[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h))


def bass_distort(frame, t, beats, bass, hihat):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2

    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx*dx + dy*dy)

    distort = np.sin(r / 8 + t * 10) * bass * 30

    map_x = (x + distort).astype(np.float32)
    map_y = (y + distort).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


# =======================================================
# 🥁 HIHAT-REACTIVE EFFECTS
# =======================================================

def hihat_flash(frame, t, beats, bass, hihat):
    if hihat < 0.2:
        return frame
    f = frame.astype(np.float32)
    f *= (1 + hihat * 2.5)
    return np.clip(f, 0, 255).astype(np.uint8)


def hihat_glitch(frame, t, beats, bass, hihat):
    if hihat < 0.15:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    glitch_amount = int(hihat * 20)

    for _ in range(glitch_amount):
        y = random.randint(0, h-3)
        band = frame[y:y+2]
        shift = random.randint(-10, 10)
        out[y:y+2] = np.roll(band, shift, axis=1)

    return out


# =======================================================
# MASTER LIST OF ALL EFFECTS
# =======================================================

VIDEO_EFFECTS = [
    # Color FX
    hue_shift,
    neon_pulse,
    solarize,

    # Glitch FX
    rgb_split,
    block_glitch,
    scanlines,

    # Distort FX
    swirl,
    wave,
    ripple,

    # Kaleidoscope / Mandala
    kaleidoscope,
    mandala_twist,

    # Pixel Sorting
    pixel_sort,
    pixel_sort_vertical,

    # Beat-reactive
    beat_glow,
    beat_rgb_shake,
    beat_zoom,
    beat_kaleido_pulse,
    beat_ripple,

    # Bass-reactive
    bass_zoom,
    bass_distort,

    # Hihat-reactive
    hihat_flash,
    hihat_glitch,
]

