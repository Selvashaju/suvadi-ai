from pathlib import Path

import pytesseract
from PIL import Image


# Tesseract installation on Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def run_ocr(image_path: str, lang: str = "tam") -> list[dict]:
    """
    Run Tamil OCR on one image.

    Returns word-level OCR evidence:
    - text
    - confidence
    - x
    - y
    - width
    - height
    """

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_file}"
        )

    # Open the image
    with Image.open(image_file) as image:
        image = image.convert("RGB")

        # Get word-level OCR data
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

    tokens = []

    for i, text in enumerate(data["text"]):

        text = text.strip()

        # Ignore empty OCR results
        if not text:
            continue

        # Convert confidence to number
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1.0

        # Bounding box
        x = int(data["left"][i])
        y = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        # Ignore invalid boxes
        if width <= 0 or height <= 0:
            continue

        token = {
            "text": text,
            "confidence": max(0.0, confidence),
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

        tokens.append(token)

    return tokens


# --------------------------------------------------
# Command-line test
# --------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python ocr_engine.py <image_path>")
        raise SystemExit(1)

    image_path = sys.argv[1]

    try:
        results = run_ocr(image_path)

        print("\nTamil OCR Results")
        print("=" * 50)

        if not results:
            print("No readable Tamil text detected.")
        else:
            for number, token in enumerate(results, start=1):
                print(
                    f"{number}. "
                    f"Text: {token['text']} | "
                    f"Confidence: {token['confidence']:.1f} | "
                    f"Box: "
                    f"({token['x']}, {token['y']}, "
                    f"{token['width']}, {token['height']})"
                )

    except Exception as error:
        print(f"OCR Error: {error}")