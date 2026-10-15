"""
Character / word accuracy of OCR on original vs each variant.

    python tools/accuracy.py samples/real_scan_01.png tests/ground_truth/real_scan_01.txt

accuracy = 1 - edit_distance / len(ground_truth)   (can go below 0 if OCR adds junk)
ZWNJ/ZWJ and extra spaces are removed before comparing.
"""
import platform
import sys
import unicodedata
from pathlib import Path

import cv2
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import preprocess_image  # noqa: E402

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return " ".join(text.split())


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def scores(ocr: str, truth: str) -> tuple[float, float]:
    o, t = normalise(ocr), normalise(truth)
    char_acc = 1 - edit_distance(o, t) / max(1, len(t))
    ow, tw = o.split(), t.split()
    word_acc = 1 - edit_distance(ow, tw) / max(1, len(tw))
    return char_acc * 100, word_acc * 100


def main(img_path, truth_path):
    truth = Path(truth_path).read_text(encoding="utf-8")
    r = preprocess_image(img_path)
    if r["status"] != "ok":
        sys.exit(f"preprocess failed: {r['error_code']}")
    images = {"original": cv2.cvtColor(r["original_rgb"], cv2.COLOR_RGB2GRAY),
              **r["variants"]}
    print(f"{'image':<11} {'char_acc%':>10} {'word_acc%':>10}")
    print("-" * 34)
    for name, img in images.items():
        text = pytesseract.image_to_string(img, lang="tam", config="--psm 6")
        c, w = scores(text, truth)
        print(f"{name:<11} {c:>10.1f} {w:>10.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
