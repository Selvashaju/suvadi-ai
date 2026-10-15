"""Run:  python -m pytest -q tests/test_preprocessing.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing as pp  # noqa: E402


def fake_page(w=1400, h=1800, angle=0.0):
    """White page with dark text-like lines (shape test, not OCR test)."""
    img = np.full((h, w, 3), 245, np.uint8)
    rng = np.random.default_rng(0)
    for y in range(150, h - 150, 60):
        x = 120
        while x < w - 200:
            ww = int(rng.integers(40, 160))
            cv2.rectangle(img, (x, y), (x + ww, y + 22), (30, 30, 30), -1)
            x += ww + 25
    if angle:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(245, 245, 245))
    return img


def png_bytes(img_rgb):
    return cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def test_three_variants_same_size():
    r = pp.preprocess_image(png_bytes(fake_page()))
    assert r["status"] == "ok"
    assert set(r["variants"]) == {"gray", "threshold", "sharpened"}
    shapes = {v.shape for v in r["variants"].values()}
    assert len(shapes) == 1
    assert shapes.pop() == r["base_rgb"].shape[:2]


def test_original_untouched():
    data = png_bytes(fake_page())
    before = pp.decode_image(data)
    r = pp.preprocess_image(data)
    assert np.array_equal(before, r["original_rgb"])
    with pytest.raises(ValueError):
        r["original_rgb"][0, 0, 0] = 0  # read-only


@pytest.mark.parametrize("angle", [-6.0, 4.0])
def test_deskew_recovers_angle(angle):
    data = png_bytes(fake_page(angle=angle))
    r = pp.preprocess_image(data, {"auto_crop": False})
    # rotating by -angle undoes a rotation of +angle
    assert abs(r["metadata"]["deskew_angle"] + angle) <= 0.75


def test_corrupt_and_wrong_type():
    assert pp.preprocess_image(b"")["error_code"] == "ERR_EMPTY_INPUT"
    assert pp.preprocess_image(b"GIF89a....")["error_code"] == "ERR_UNSUPPORTED_TYPE"
    broken = b"\x89PNG\r\n\x1a\n" + b"garbage" * 10
    assert pp.preprocess_image(broken)["error_code"] == "ERR_DECODE_FAILED"


def test_deterministic():
    data = png_bytes(fake_page(angle=3))
    a = pp.preprocess_image(data)["variants"]
    b = pp.preprocess_image(data)["variants"]
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_crop_and_map_back():
    data = png_bytes(fake_page(angle=5))
    r = pp.preprocess_image(data, {"auto_crop": False})
    box = {"x": 200, "y": 300, "width": 120, "height": 40}
    crop = pp.crop_word(r["base_rgb"], box, pad=0)
    assert crop.shape[:2] == (40, 120)
    corners = pp.map_box_to_original(box, r["metadata"])
    assert len(corners) == 4


def test_heatmap_same_size():
    r = pp.preprocess_image(png_bytes(fake_page()))
    toks = [{"bounding_box": {"x": 10, "y": 10, "width": 50, "height": 20}, "evidence_score": 70}]
    assert pp.heatmap_overlay(r["base_rgb"], toks).shape == r["base_rgb"].shape
