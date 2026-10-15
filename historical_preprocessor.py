from pathlib import Path
import cv2


def preprocess_historical_image(
    image_path: str,
    output_dir: str = "output",
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # Resize for better OCR visibility
    resized = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    # Mild denoising
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    # Variant 1: enhanced grayscale
    gray_variant = denoised

    # Variant 2: adaptive threshold
    threshold_variant = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    # Variant 3: sharpened
    blurred = cv2.GaussianBlur(
        denoised,
        (0, 0),
        2,
    )

    sharpened_variant = cv2.addWeighted(
        denoised,
        1.8,
        blurred,
        -0.8,
        0,
    )

    variants = {
        "gray": gray_variant,
        "threshold": threshold_variant,
        "sharpened": sharpened_variant,
    }

    saved_files = {}

    for name, variant in variants.items():

        output_path = (
            output_dir
            / f"{image_path.stem}_{name}.png"
        )

        if not cv2.imwrite(
            str(output_path),
            variant,
        ):
            raise RuntimeError(
                f"Could not save {output_path}"
            )

        saved_files[name] = str(output_path)

    return saved_files


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python historical_preprocessor.py <image>"
        )
        raise SystemExit(1)

    try:

        results = preprocess_historical_image(
            sys.argv[1]
        )

        print("\nHistorical preprocessing complete")
        print("=" * 50)

        for name, path in results.items():
            print(f"{name:12} -> {path}")

    except Exception as error:
        print(f"Preprocessing Error: {error}")
