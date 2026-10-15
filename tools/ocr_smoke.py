"""
Quick check: does preprocessing help Tamil OCR?
    python tools/ocr_smoke.py samples/sim_faded.png
"""
import platform
import sys
from pathlib import Path

import cv2
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import preprocess_image  # noqa: E402

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LANG = "tam"


def ocr_stats(img):
    d = pytesseract.image_to_data(img, lang=LANG, config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(d["text"], d["conf"])
             if t.strip() and float(c) >= 0]
    mean = sum(c for _, c in words) / len(words) if words else 0.0
    return len(words), mean, " ".join(t for t, _ in words[:12])


def main(path):
    r = preprocess_image(path)
    if r["status"] != "ok":
        sys.exit(f"preprocess failed: {r['error_code']}")
    images = {"original": cv2.cvtColor(r["original_rgb"], cv2.COLOR_RGB2GRAY),
              **r["variants"]}
    print(f"{'image':<11} {'words':>6} {'mean_conf':>10}  first words")
    print("-" * 70)
    for name, img in images.items():
        n, conf, sample = ocr_stats(img)
        print(f"{name:<11} {n:>6} {conf:>10.1f}  {sample}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/sim_faded.png")
