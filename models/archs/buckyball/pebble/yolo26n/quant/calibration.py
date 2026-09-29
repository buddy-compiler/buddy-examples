import numpy as np
from PIL import Image


def load_calibration_images(paths, size):
    if not paths or size <= 0:
        raise ValueError("calibration requires images and a positive input size")
    samples = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
        height, width = rgb.shape[:2]
        ratio = min(np.float32(size) / height, np.float32(size) / width)
        out_h = int(np.floor(height * ratio + 0.5))
        out_w = int(np.floor(width * ratio + 0.5))
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"image aspect ratio produces an empty resize: {path}")
        top, left = (size - out_h) // 2, (size - out_w) // 2
        y = np.arange(out_h, dtype=np.float32) * (np.float32(height) / out_h)
        x = np.arange(out_w, dtype=np.float32) * (np.float32(width) / out_w)
        yl, yh = np.floor(y).astype(int), np.minimum(np.ceil(y).astype(int), height - 1)
        xl, xh = np.floor(x).astype(int), np.minimum(np.ceil(x).astype(int), width - 1)
        wy = (y - np.floor(y))[:, None, None]
        wx = (x - np.floor(x))[None, :, None]
        a, b = rgb[yl[:, None], xl], rgb[yh[:, None], xl]
        c, d = rgb[yl[:, None], xh], rgb[yh[:, None], xh]
        # Match the guest DIP corner ordering and FP32 arithmetic exactly.
        resized = (a * ((1 - wx) * (1 - wy)) + b * (wx * (1 - wy))) + (
            c * (wy * (1 - wx)) + d * (wx * wy)
        )
        sample = np.full((3, size, size), np.float32(114) / 255, dtype=np.float32)
        sample[:, top : top + out_h, left : left + out_w] = resized.transpose(2, 0, 1)
        samples.append(sample)
    return np.stack(samples)
