"""
SUVADI AI - Member 1: Image preprocessing.

Contract (shared with Member 2 / Member 3 / Member 4):

    result = preprocess_image(source, settings=None, page_id="page_001")

    result["status"]        "ok" | "error"
    result["error_code"]    None | one of ERROR_CODES (Member 3 maps to Tamil)
    result["original_rgb"]  untouched RGB uint8 array (NEVER modified)
    result["base_rgb"]      geometry-corrected RGB (crop + deskew). All variants
                            and all OCR boxes live in THIS coordinate system.
    result["variants"]      {"gray": HxW uint8, "threshold": HxW uint8,
                             "sharpened": HxW uint8}  -- identical H and W
    result["metadata"]      width, height, deskew_angle, transform matrix,
                            settings, quality stats, warnings, source sha256

Crops for the review screen must be taken from base_rgb (or any variant),
NOT from original_rgb, because OCR boxes are in base coordinates.
Use map_box_to_original() if a crop of the untouched original is needed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Status codes. English only in code; Member 3 maps these to Tamil text in
# locale/ui_ta.json. Never show these strings directly in the UI.
# --------------------------------------------------------------------------
ERROR_CODES = {
    "ERR_EMPTY_INPUT",
    "ERR_UNSUPPORTED_TYPE",
    "ERR_DECODE_FAILED",
    "ERR_TOO_SMALL",
}
WARNING_CODES = {
    "WARN_LOW_RESOLUTION",
    "WARN_RESIZED_FOR_SPEED",
    "WARN_CROP_FALLBACK_FULL_IMAGE",
    "WARN_DESKEW_SKIPPED_UNSURE",
    "WARN_DESKEW_ANGLE_LIMIT",
    "WARN_LOW_CONTRAST",
    "WARN_BLURRY",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "max_side": 3000,          # bigger images are resized (processing copy only)
    "min_side": 400,           # below this we refuse
    "low_res_side": 1000,      # below this we warn
    "auto_crop": True,         # detect page contour; falls back to full image
    "auto_deskew": True,
    "manual_rotation": None,   # degrees; overrides auto deskew when set
    "max_deskew_angle": 15.0,  # never rotate more than this automatically
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "median_kernel": 3,        # keep 3: 5 can erase Tamil pulli dots
    "threshold_method": "adaptive",  # "adaptive" | "otsu"
    "threshold_block": None,   # None = auto from image size (odd number)
    "threshold_c": 10,
    "sharpen_amount": 0.6,     # light unsharp mask only
    "sharpen_sigma": 1.0,
}

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPG_MAGIC = b"\xff\xd8\xff"


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def read_source_bytes(source: str | Path | bytes | bytearray) -> bytes:
    """Accept a file path, raw bytes, or a Streamlit UploadedFile."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):  # streamlit UploadedFile / BytesIO
        return source.getvalue()
    return Path(source).read_bytes()


def detect_image_type(data: bytes) -> str | None:
    if data.startswith(_PNG_MAGIC):
        return "png"
    if data.startswith(_JPG_MAGIC):
        return "jpg"
    return None


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode bytes to an RGB uint8 array. Returns None on failure."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Quality statistics
# --------------------------------------------------------------------------
def quality_stats(gray: np.ndarray) -> dict[str, float]:
    return {
        "brightness_mean": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "sharpness_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


# --------------------------------------------------------------------------
# Geometry: resize, crop, deskew
# --------------------------------------------------------------------------
def limit_size(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb, 1.0
    out = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_page_quad(rgb: np.ndarray) -> np.ndarray | None:
    """Find a 4-corner page outline covering 30-98% of the image, else None."""
    h, w = rgb.shape[:2]
    small_scale = 800 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(rgb, None, fx=small_scale, fy=small_scale) if small_scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        ratio = cv2.contourArea(approx) / area_total
        if len(approx) == 4 and 0.30 <= ratio <= 0.98:
            return _order_corners(approx) / small_scale
    return None


def warp_to_quad(rgb: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    out = cv2.warpPerspective(rgb, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, m


def estimate_skew_angle(gray: np.ndarray, search: float = 15.0,
                        step: float = 0.25) -> tuple[float, float]:
    """
    Projection-profile deskew: try angles, pick the one where text rows are
    sharpest (max variance of row sums). Avoids the OpenCV minAreaRect angle
    convention trap. Returns (angle_degrees, confidence 0..1).
    Positive angle = rotate counter-clockwise to straighten.
    """
    h, w = gray.shape
    scale = 1000 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    # adaptive (not Otsu): a camera shadow must not be counted as ink
    ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    if ink.mean() < 1.0:  # almost no ink: nothing to align
        return 0.0, 0.0
    sh, sw = ink.shape
    centre = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(ink, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1, dtype=np.float64)))

    angles = np.arange(-search, search + step / 2, step)
    scores = np.array([score(a) for a in angles])
    best = int(np.argmax(scores))
    # confidence: how much the best angle stands out from the median
    median = float(np.median(scores)) or 1.0
    conf = float(np.clip((scores[best] / median - 1.0) / 2.0, 0.0, 1.0))
    return round(float(angles[best]), 2), round(conf, 3)


def rotate_keep_all(rgb: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate without cutting corners; new area filled with paper colour."""
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    paper = tuple(int(v) for v in np.median(rgb.reshape(-1, 3), axis=0))
    out = cv2.warpAffine(rgb, m, (nw, nh), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    return out, m


# --------------------------------------------------------------------------
# Tone: grayscale, CLAHE, denoise, threshold, sharpen
# --------------------------------------------------------------------------
def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def apply_clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def median_denoise(gray: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1)  # must be odd
    return cv2.medianBlur(gray, k) if k > 1 else gray.copy()


def auto_block_size(gray: np.ndarray) -> int:
    b = int(min(gray.shape) / 40)
    b = int(np.clip(b, 15, 61))
    return b if b % 2 else b + 1


def binarize(gray: np.ndarray, method: str, block: int | None, c: int) -> np.ndarray:
    if method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    block = block or auto_block_size(gray)
    block = block if block % 2 else block + 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def unsharp(gray: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def _error(code: str, meta: dict) -> dict:
    return {"status": "error", "error_code": code, "original_rgb": None,
            "base_rgb": None, "variants": {}, "metadata": meta}


def preprocess_image(source, settings: dict | None = None,
                     page_id: str = "page") -> dict:
    t0 = time.perf_counter()
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    warnings: list[str] = []
    meta: dict[str, Any] = {"page_id": page_id, "settings": s, "warnings": warnings}

    try:
        data = read_source_bytes(source)
    except (OSError, TypeError):
        return _error("ERR_DECODE_FAILED", meta)
    if not data:
        return _error("ERR_EMPTY_INPUT", meta)

    meta["source_sha256"] = sha256_bytes(data)
    kind = detect_image_type(data)
    if kind is None:
        return _error("ERR_UNSUPPORTED_TYPE", meta)
    meta["source_type"] = kind

    original = decode_image(data)
    if original is None:
        return _error("ERR_DECODE_FAILED", meta)
    original.setflags(write=False)  # hard guarantee: nobody can modify it
    oh, ow = original.shape[:2]
    meta["original_size"] = {"width": ow, "height": oh}

    if min(oh, ow) < s["min_side"]:
        return {**_error("ERR_TOO_SMALL", meta), "original_rgb": original}
    if min(oh, ow) < s["low_res_side"]:
        warnings.append("WARN_LOW_RESOLUTION")

    # 1. processing copy (resized if huge)
    work, scale = limit_size(original.copy(), s["max_side"])
    if scale < 1.0:
        warnings.append("WARN_RESIZED_FOR_SPEED")
    meta["resize_scale"] = round(scale, 4)

    # 2. page crop
    meta["perspective_matrix"] = None
    if s["auto_crop"]:
        quad = find_page_quad(work)
        if quad is not None:
            work, pm = warp_to_quad(work, quad)
            meta["perspective_matrix"] = pm.tolist()
        else:
            warnings.append("WARN_CROP_FALLBACK_FULL_IMAGE")

    # 3. deskew
    angle, conf = 0.0, 0.0
    if s["manual_rotation"] is not None:
        angle, conf = float(s["manual_rotation"]), 1.0
    elif s["auto_deskew"]:
        angle, conf = estimate_skew_angle(to_gray(work), search=s["max_deskew_angle"])
        if abs(angle) >= s["max_deskew_angle"] - 0.01:
            warnings.append("WARN_DESKEW_ANGLE_LIMIT")
            angle = 0.0
        elif conf < 0.05:
            warnings.append("WARN_DESKEW_SKIPPED_UNSURE")
            angle = 0.0
    meta["deskew_angle"] = angle
    meta["deskew_confidence"] = conf
    meta["rotation_matrix"] = None
    if abs(angle) >= 0.25:
        work, rm = rotate_keep_all(work, angle)
        meta["rotation_matrix"] = rm.tolist()

    base_rgb = work
    bh, bw = base_rgb.shape[:2]
    meta["width"], meta["height"] = bw, bh

    # 4. tone variants (all share base_rgb geometry)
    gray_raw = to_gray(base_rgb)
    stats = quality_stats(gray_raw)
    meta["quality"] = stats
    if stats["contrast_std"] < 25:
        warnings.append("WARN_LOW_CONTRAST")
    if stats["sharpness_laplacian_var"] < 50:
        warnings.append("WARN_BLURRY")

    gray = apply_clahe(gray_raw, s["clahe_clip"], s["clahe_tile"])
    denoised = median_denoise(gray, s["median_kernel"])
    block = s["threshold_block"] or auto_block_size(denoised)
    meta["threshold_block_used"] = block
    threshold = binarize(denoised, s["threshold_method"], block, s["threshold_c"])
    sharpened = unsharp(denoised, s["sharpen_amount"], s["sharpen_sigma"])

    variants = {"gray": gray, "threshold": threshold, "sharpened": sharpened}
    assert all(v.shape == (bh, bw) for v in variants.values())

    meta["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"status": "ok", "error_code": None, "original_rgb": original,
            "base_rgb": base_rgb, "variants": variants, "metadata": meta}


# --------------------------------------------------------------------------
# Helpers for Member 2 (OCR boxes) and Member 3 (review screen / heatmap)
# --------------------------------------------------------------------------
def crop_word(image: np.ndarray, box: dict, pad: int = 6) -> np.ndarray:
    """box = {"x","y","width","height"} in base coordinates."""
    h, w = image.shape[:2]
    x0 = max(0, int(box["x"]) - pad)
    y0 = max(0, int(box["y"]) - pad)
    x1 = min(w, int(box["x"] + box["width"]) + pad)
    y1 = min(h, int(box["y"] + box["height"]) + pad)
    return image[y0:y1, x0:x1].copy()


def map_box_to_original(box: dict, metadata: dict) -> list[tuple[float, float]]:
    """Map a base-coordinate box back to the untouched original (4 corners)."""
    x, y, bw, bh = box["x"], box["y"], box["width"], box["height"]
    pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float64)
    if metadata.get("rotation_matrix") is not None:
        m = np.vstack([np.array(metadata["rotation_matrix"]), [0, 0, 1]])
        pts = cv2.transform(pts[None], np.linalg.inv(m)[:2])[0]
    if metadata.get("perspective_matrix") is not None:
        pm = np.linalg.inv(np.array(metadata["perspective_matrix"]))
        pts = cv2.perspectiveTransform(pts[None], pm)[0]
    pts = pts / metadata.get("resize_scale", 1.0)
    return [(round(float(px), 1), round(float(py), 1)) for px, py in pts]


HEAT_COLOURS = {"green": (46, 160, 67), "amber": (230, 160, 0), "red": (214, 40, 40)}


def score_colour(score: float) -> str:
    return "green" if score >= 85 else "amber" if score >= 60 else "red"


def heatmap_overlay(base_rgb: np.ndarray, tokens: list[dict], alpha: float = 0.35) -> np.ndarray:
    """tokens: [{"bounding_box": {...}, "evidence_score": 0-100}, ...]"""
    overlay = base_rgb.copy()
    for t in tokens:
        b = t["bounding_box"]
        c = HEAT_COLOURS[score_colour(t.get("evidence_score", 0))]
        p0 = (int(b["x"]), int(b["y"]))
        p1 = (int(b["x"] + b["width"]), int(b["y"] + b["height"]))
        cv2.rectangle(overlay, p0, p1, c, -1)
    out = cv2.addWeighted(overlay, alpha, base_rgb, 1 - alpha, 0)
    for t in tokens:
        b = t["bounding_box"]
        c = HEAT_COLOURS[score_colour(t.get("evidence_score", 0))]
        cv2.rectangle(out, (int(b["x"]), int(b["y"])),
                      (int(b["x"] + b["width"]), int(b["y"] + b["height"])), c, 2)
    return out


def to_png_bytes(image: np.ndarray) -> bytes:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""


def public_metadata(metadata: dict) -> dict:
    """JSON-safe metadata for the audit record (Member 4)."""
    return {k: v for k, v in metadata.items()}
