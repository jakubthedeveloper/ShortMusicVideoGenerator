import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------
# INTERNAL UTILS
# ---------------------------------------------------------

def _to_cuda_tensor(frame):
    """Convert numpy RGB frame -> torch CUDA tensor CHW float32."""
    if isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame).float() / 255.0
        t = t.permute(2, 0, 1).contiguous()    # HWC → CHW
        return t.unsqueeze(0).cuda()           # add batch dimension
    return frame


def _to_numpy_frame(tensor):
    """Convert CUDA tensor -> numpy RGB frame."""
    t = tensor.squeeze(0).detach().cpu()
    t = (t.clamp(0, 1) * 255).byte()
    t = t.permute(1, 2, 0).contiguous()        # CHW → HWC
    return t.numpy()


def _polar_grid(h, w, device):
    """Create polar coordinate maps for kaleidoscope segmentation."""
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    cx, cy = w // 2, h // 2
    x = xx - cx
    y = yy - cy
    angle = torch.atan2(y, x)  # radians
    dist = torch.sqrt(x*x + y*y)
    return angle, dist


def _rotate_tensor(img, angle_deg):
    """Rotate tensor using affine grid + grid_sample."""
    B, C, H, W = img.shape
    angle_rad = torch.tensor([angle_deg * np.pi / 180], device=img.device)

    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)

    # 2D rotation matrix
    M = torch.zeros((1, 2, 3), device=img.device)
    M[0, 0, 0] = cos
    M[0, 0, 1] = -sin
    M[0, 1, 0] = sin
    M[0, 1, 1] = cos

    grid = F.affine_grid(M, img.size(), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False)


def _mirror_tensor(img):
    """Flip horizontally using tensor slicing."""
    return img.flip(dims=[3])  # flip width dimension


# ---------------------------------------------------------
# BASE KALEIDOSCOPE FUNCTION (CUDA)
# ---------------------------------------------------------

def kaleidoscope_mirrored_cuda(frame, t, beats, bass, hihat):
    """
    Real mirrored kaleidoscope with dynamic segments (reacts to bass/hihat).
    """
    try:
        tensor = _to_cuda_tensor(frame)
        B, C, H, W = tensor.shape
        device = tensor.device

        # dynamic segment count
        segments = int(6 + bass * 8 + hihat * 4 + 2 * np.sin(t * 0.7))
        segments = max(4, min(segments, 32))

        angle, dist = _polar_grid(H, W, device)
        angle = (angle + np.pi * 2) % (np.pi * 2)

        step = (np.pi * 2) / segments
        output = torch.zeros_like(tensor)

        for i in range(segments):
            a0 = i * step
            a1 = a0 + step

            mask = (angle >= a0) & (angle < a1)

            # rotate so wedge starts upright
            rotated = _rotate_tensor(tensor, -a0 * 180 / np.pi)

            # mirror every second wedge
            if i % 2 == 1:
                rotated = _mirror_tensor(rotated)

            # apply wedge mask
            for c in range(C):
                output[0, c][mask] = rotated[0, c][mask]

        return _to_numpy_frame(output)

    except Exception as e:
        print("[CUDA kaleidoscope_mirrored FAILED]", e)
        return frame


# ---------------------------------------------------------
# DYNAMIC KALEIDOSCOPE
# ---------------------------------------------------------

def kaleidoscope_dynamic_cuda(frame, t, beats, bass, hihat):
    return kaleidoscope_mirrored_cuda(frame, t, beats, bass, hihat)


# ---------------------------------------------------------
# LIQUID KALEIDOSCOPE
# ---------------------------------------------------------

def kaleidoscope_liquid_cuda(frame, t, beats, bass, hihat):
    """Liquid glass warping before kaleidoscope."""
    try:
        tensor = _to_cuda_tensor(frame)
        B, C, H, W = tensor.shape
        device = tensor.device

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        r = torch.sqrt(xx*xx + yy*yy)
        strength = 0.05 + bass * 0.15
        swirl = torch.sin(r * 10 + t * 3) * strength

        map_x = xx + xx * swirl
        map_y = yy + yy * swirl

        grid = torch.stack((map_x, map_y), dim=-1).unsqueeze(0)
        warped = F.grid_sample(tensor, grid, align_corners=False)

        # feed into mirrored kaleidoscope
        img_np = _to_numpy_frame(warped)
        return kaleidoscope_mirrored_cuda(img_np, t, beats, bass, hihat)

    except Exception as e:
        print("[CUDA kaleidoscope_liquid FAILED]", e)
        return frame


# ---------------------------------------------------------
# 3D CURVED KALEIDOSCOPE (LENS DISTORT)
# ---------------------------------------------------------

def kaleidoscope_3d_cuda(frame, t, beats, bass, hihat):
    try:
        tensor = _to_cuda_tensor(frame)
        B, C, H, W = tensor.shape
        device = tensor.device

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        r2 = xx*xx + yy*yy
        k = 0.2 + bass * 0.4
        scale = 1 + r2 * k

        map_x = xx / scale
        map_y = yy / scale

        grid = torch.stack((map_x, map_y), dim=-1).unsqueeze(0)
        curved = F.grid_sample(tensor, grid, align_corners=False)

        img_np = _to_numpy_frame(curved)
        return kaleidoscope_mirrored_cuda(img_np, t, beats, bass, hihat)

    except Exception as e:
        print("[CUDA kaleidoscope_3d FAILED]", e)
        return frame


# ---------------------------------------------------------
# FRACTAL KALEIDOSCOPE
# ---------------------------------------------------------

def kaleidoscope_fractal_cuda(frame, t, beats, bass, hihat):
    try:
        tensor = _to_cuda_tensor(frame)
        B, C, H, W = tensor.shape

        zoom = 1 + 0.2 * np.sin(t*1.5 + bass*5)
        rot = t * 20

        rotated = _rotate_tensor(tensor, rot)
        zoomed = F.interpolate(rotated, scale_factor=zoom, mode="bilinear",
                               align_corners=False)

        # center crop back to original size
        zh, zw = zoomed.shape[2:]
        dh = (zh - H) // 2
        dw = (zw - W) // 2
        cropped = zoomed[:, :, dh:dh+H, dw:dw+W]

        # blend fractal layers
        blend = tensor * 0.5 + cropped * 0.5

        img_np = _to_numpy_frame(blend)
        return kaleidoscope_mirrored_cuda(img_np, t, beats, bass, hihat)

    except Exception as e:
        print("[CUDA kaleidoscope_fractal FAILED]", e)
        return frame

