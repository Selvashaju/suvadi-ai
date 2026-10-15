# SUVADI AI – Member 2 Tamil OCR & Evidence Engine

## Purpose

Member 2 provides the Tamil OCR, historical-image preprocessing, multi-variant evidence comparison and structured OCR output used by the SUVADI AI UI.

## Modules

### ocr_engine.py

Main function:

    run_ocr(image_path, lang="tam")

Input:
- image_path: path to an image
- lang: Tesseract language code, default `tam`

Output:
A list of OCR tokens containing:

- text
- confidence
- x
- y
- width
- height

Example:

    {
        "text": "உம்மை",
        "confidence": 93.0,
        "x": 285,
        "y": 288,
        "width": 104,
        "height": 28
    }

## Historical preprocessing

### historical_preprocessor.py

Main function:

    preprocess_historical_image(image_path)

Produces:

- Gray
- Threshold
- Sharpened

Each variant is returned with its generated image path.

## Evidence Engine

### evidence_engine.py

Main function:

    compare_variants(ocr_results)

The engine compares OCR results from the available preprocessing variants.

Evidence result contains:

- text
- confidence
- agreement
- score
- status
- alternatives

The application uses three evidence states:

### HIGH

Strong OCR confidence and strong agreement between variants.

### REVIEW

The result has uncertainty or disagreement between variants and should be verified by a human reviewer.

### UNREADABLE

Low confidence and/or strong disagreement. The result should not be automatically accepted.

## Unicode normalization

OCR text is normalized before comparison.

The normalization process includes:

- Unicode NFC normalization
- Removal of U+200C Zero Width Non-Joiner
- Removal of U+200D Zero Width Joiner

This prevents invisible Unicode differences from affecting evidence comparison.

## Pipeline

### run_pipeline.py

Execution:

    python run_pipeline.py "<image_path>"

Workflow:

    Image
      ↓
    Historical preprocessing
      ↓
    Gray / Threshold / Sharpened
      ↓
    Tamil OCR
      ↓
    Evidence comparison
      ↓
    Structured evidence_results.json

## Tesseract setup

Install Tesseract OCR on the system.

Tamil language support is required.

The Tesseract language code for Tamil is:

    tam

Verify installation:

    tesseract --version

Verify Tamil language data:

    tesseract --list-langs

The output should contain:

    tam

## Python dependencies

Install:

    python -m pip install -r requirements.txt

## Testing

Run the test suite:

    python -m pytest -q

## Validation

Member 2 validation was performed using the team's real scan, ground truth and simulated degradation samples.

Clean real scan:

    Character accuracy: 97.9%
    Word accuracy: 90.0%

Faded:

    Character accuracy: 98.5%
    Word accuracy: 91.2%

Noisy:

    Best variant: Sharpened
    Character accuracy: 93.1%
    Word accuracy: 81.2%

Shadow:

    Best variant: Threshold
    Character accuracy: 95.0%
    Word accuracy: 82.5%

Rotated:

    Best variant: Gray
    Character accuracy: 97.7%
    Word accuracy: 90.0%

The results show that different degradation conditions can favor different preprocessing variants. Therefore the multi-variant evidence approach is retained.

## Integration sample

Real sample:

    samples/member2/evidence_results.json

Member 3 can use this file to test UI rendering before connecting live OCR output.

## Integration notes

Do not treat OCR output as automatically verified historical text.

Recommended UI flow:

    OCR result
      ↓
    Evidence state
      ↓
    Human review
      ↓
    Verified text
      ↓
    Archive

Bounding-box coordinates use:

    x
    y
    width
    height

The review/storage layer should preserve the original OCR value separately from the human-corrected value.
