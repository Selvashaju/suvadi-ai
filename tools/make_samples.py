"""
Create SIMULATED damaged copies of one clean page (labelled honestly).

    python tools/make_samples.py samples/clean_page.png

Writes samples/sim_faded.png, sim_rotated.png, sim_shadow.png, sim_noisy.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

rng = np.random.default_rng(42)  # fixed seed: same samples every run


def faded(img):
    out = img.astype(np.float32)
    out = 255 - (255 - out) * 0.35          # ink becomes pale
    out = out * np.array([0.93, 0.88, 0.78])  # yellowed paper (BGR)
    return np.clip(out, 0, 255).astype(np.uint8)


def rotated(img, angle=6.0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def shadowed(img):
    h, w = img.shape[:2]
    grad = np.tile(np.linspace(0.45, 1.0, w, dtype=np.float32), (h, 1))[..., None]
    return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def noisy(img):
    out = img.astype(np.float32) + rng.normal(0, 22, img.shape)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)
    mask = rng.random(img.shape[:2]) < 0.01
    out[mask] = 0  # specks
    return out


def main(path: str):
    src = Path(path)
    img = cv2.imread(str(src))
    if img is None:
        sys.exit(f"cannot read {src}")
    out_dir = src.parent
    for name, fn in [("faded", faded), ("rotated", rotated),
                     ("shadow", shadowed), ("noisy", noisy)]:
        dst = out_dir / f"sim_{name}.png"
        cv2.imwrite(str(dst), fn(img))
        print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/clean_page.png")
