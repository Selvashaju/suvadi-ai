"""Member 3 adapter: aligned Member 1 images -> Member 2 OCR and evidence.

Keeps teammates' modules unchanged.  OCR is optionally run on an enlarged
temporary COPY of each image when small words appear undersampled; boxes are
mapped back into Member 1's unscaled base_rgb pixel coordinates.  The original
image, OCR raw text, and previously stored human review decisions are untouched.
"""
from __future__ import annotations

import logging
import statistics
import tempfile
from pathlib import Path

import cv2

from evidence_engine import compare_variants, find_nearby_token, tokens_match
from ocr_engine import run_ocr
from preprocessing import to_png_bytes

VARIANT_ORDER = ("gray", "threshold", "sharpened")
# Avoid creating very large temporary files on the judging laptop.
MAX_OCR_LONG_SIDE = 4200
TARGET_TOKEN_HEIGHT = 27
MAX_UPSCALE = 2.7


# These are REVIEW FLAGS, not automatic OCR corrections or deletions.
# False positives are possible, including punctuation, Tamil pulli, and
# historical page numbers. Every candidate remains available to a reviewer.
def review_noise_reasons(token: dict) -> list[str]:
    """Return conservative UI triage flags for saved or new OCR token dicts."""
    box = token.get("bounding_box") or token
    try:
        width = max(0, int(box.get("width", 0)))
        height = max(0, int(box.get("height", 0)))
        confidence = float(token.get("ocr_confidence", token.get("confidence", 0)))
    except (ValueError, TypeError):
        return []

    raw = str(token.get("raw_value", token.get("text", ""))).strip()
    reasons = []
    if width and height and (min(width, height) <= 3 or width * height < 16):
        reasons.append("tiny_box")
    # Numerals can carry historical information, so never flag ALL numbers.
    if raw and not any(ch.isalnum() for ch in raw):
        reasons.append("punctuation_only")
    if raw and len(raw) <= 2 and confidence < 10:
        reasons.append("short_low_confidence")
    return reasons


def _tamil_quality(tokens: list[dict]) -> tuple[int, float, int]:
    """Select a reading using OCR evidence, not by guessing Tamil words."""
    tamil = [
        t for t in tokens
        if any("\u0b80" <= ch <= "\u0bff" for ch in str(t.get("text", "")))
    ]
    confident = [t for t in tamil if float(t.get("confidence", 0)) >= 15.0]
    mean_conf = (
        sum(float(t["confidence"]) for t in confident) / len(confident)
        if confident else 0.0
    )
    # Prioritize the number of genuinely confident Tamil words, then average
    # confidence. Zero-confidence words are retained for manual review.
    return len(confident), round(mean_conf, 3), len(tamil)


def _select_upscale(tokens: list[dict], width: int, height: int) -> float:
    heights = [int(t.get("height", 0)) for t in tokens if int(t.get("height", 0)) > 0]
    median_h = statistics.median(heights) if heights else 12
    scale = TARGET_TOKEN_HEIGHT / max(1.0, median_h)
    if not any(float(t.get("confidence", 0)) >= 15.0 for t in tokens):
        scale = max(scale, 1.8)
    limit = MAX_OCR_LONG_SIDE / max(width, height)
    scale = min(MAX_UPSCALE, scale, limit)
    return round(scale, 4) if scale >= 1.35 else 1.0


def _map_to_base(tokens: list[dict], scale_x: float, scale_y: float,
                 base_width: int, base_height: int) -> list[dict]:
    """Convert enlarged-image word boxes back to original base coordinates."""
    mapped = []
    for token in tokens:
        x = int(token["x"])
        y = int(token["y"])
        w = int(token["width"])
        h = int(token["height"])
        # Preserve a covering rectangle, even when division leaves fractions.
        import math
        x0 = max(0, min(base_width, math.floor(x / scale_x)))
        y0 = max(0, min(base_height, math.floor(y / scale_y)))
        x1 = max(0, min(base_width, math.ceil((x + w) / scale_x)))
        y1 = max(0, min(base_height, math.ceil((y + h) / scale_y)))
        if x1 <= x0 or y1 <= y0:
            continue
        mapped.append({
            **token,
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
        })
    return mapped


def _run_variant_ocr(name: str, variant, scratch: Path) -> list[dict]:
    h, w = variant.shape[:2]
    base_path = scratch / f"{name}_base.png"
    base_path.write_bytes(to_png_bytes(variant))
    original_tokens = run_ocr(str(base_path), lang="tam")

    factor = _select_upscale(original_tokens, w, h)
    if factor == 1.0:
        return original_tokens

    new_w, new_h = round(w * factor), round(h * factor)
    if new_w <= w or new_h <= h:
        return original_tokens

    scaled = cv2.resize(
        variant, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )
    scaled_path = scratch / f"{name}_scaled.png"
    scaled_path.write_bytes(to_png_bytes(scaled))
    enlarged_tokens = run_ocr(str(scaled_path), lang="tam")

    # Use ACTUAL width/height ratios, not the requested rounded scale.
    remapped = _map_to_base(
        enlarged_tokens, new_w / w, new_h / h, w, h
    )
    if _tamil_quality(remapped) > _tamil_quality(original_tokens):
        logging.info(
            "SUVADI OCR: %s selected scaled pass (factor %.2f); %d words",
            name, factor, len(remapped)
        )
        return remapped
    return original_tokens


def build_review_tokens(preprocess_result: dict, page_key: str) -> tuple[list[dict], dict[str, int]]:
    if preprocess_result.get("status") != "ok":
        raise ValueError("Member 1 preprocessing did not complete successfully")

    variants = preprocess_result["variants"]
    base_h, base_w = preprocess_result["base_rgb"].shape[:2]
    ocr_results: dict[str, list[dict]] = {}

    with tempfile.TemporaryDirectory(prefix="suvadi_ocr_") as directory:
        scratch = Path(directory)
        for name in VARIANT_ORDER:
            if name not in variants:
                continue
            variant = variants[name]
            if variant.shape[:2] != (base_h, base_w):
                raise ValueError(f"Variant {name} has misaligned dimensions")
            ocr_results[name] = _run_variant_ocr(name, variant, scratch)

    if not ocr_results:
        raise ValueError("No image variants were provided")

    counts = {name: len(tokens) for name, tokens in ocr_results.items()}
    valid = [name for name, tokens in ocr_results.items() if tokens]
    if not valid:
        return [], counts

    # Member 2 compare_variants() uses the FIRST nonempty image as positional
    # reference. Prefer the strongest OCR pass, rather than always using gray.
    reference_variant = max(valid, key=lambda name: _tamil_quality(ocr_results[name]))
    ordered = {
        reference_variant: ocr_results[reference_variant],
        **{n: ocr_results[n] for n in VARIANT_ORDER
           if n != reference_variant and n in ocr_results}
    }

    scored = compare_variants(ordered)
    reference_tokens = ordered[reference_variant]
    if len(scored) != len(reference_tokens):
        raise ValueError("OCR evidence result count differs from reference tokens")

    records = []
    for index, (reference, result) in enumerate(zip(reference_tokens, scored), start=1):
        box = {name: int(reference[name]) for name in ("x", "y", "width", "height")}
        sources = [reference_variant]
        for variant_name, candidates in ordered.items():
            if variant_name == reference_variant or not candidates:
                continue
            nearby = find_nearby_token(reference, candidates)
            if nearby is not None and tokens_match(reference, nearby):
                sources.append(variant_name)
        record = {
            "page_id": page_key,
            "token_id": f"t{index:05d}",
            "raw_value": result.text,
            "text": result.text,
            "original_value": result.text,
            "ocr_confidence": float(result.confidence),
            "agreement": float(result.agreement),
            "evidence_score": float(result.score),
            "evidence_state": result.status,
            "status": "review",  # human review still required, even for HIGH
            "review_action": None,
            "final_value": None,
            "reviewed_at": None,
            "bounding_box": box,
            "source_variant": reference_variant,
            "source_variants": sources,
            "alternatives": list(result.alternatives),
        }
        record["review_noise_reasons"] = review_noise_reasons(record)
        records.append(record)

    return records, counts
