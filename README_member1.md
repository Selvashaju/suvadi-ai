# Member 1 - Image preprocessing (SUVADI AI)

## Mac setup (one time)
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Check
    python -m pytest -q tests/test_preprocessing.py     # 8 tests should pass
    python tools/make_samples.py samples/clean_page.png  # makes sim_* damaged copies
    python tools/preview.py samples/sim_faded.png        # opens side-by-side preview

## Contract for teammates
    from preprocessing import preprocess_image
    r = preprocess_image(uploaded_file_or_path_or_bytes, settings, page_id="p1")
    r["status"], r["error_code"]          -> Member 3 maps codes to Tamil
    r["variants"]["gray"|"threshold"|"sharpened"]  -> Member 2 runs OCR on each
    r["base_rgb"]                          -> crops + heatmap (OCR boxes live here)
    r["metadata"]                          -> Member 4 audit record

Error codes:   ERR_EMPTY_INPUT, ERR_UNSUPPORTED_TYPE, ERR_DECODE_FAILED, ERR_TOO_SMALL
Warning codes: WARN_LOW_RESOLUTION, WARN_RESIZED_FOR_SPEED, WARN_CROP_FALLBACK_FULL_IMAGE,
               WARN_DESKEW_SKIPPED_UNSURE, WARN_DESKEW_ANGLE_LIMIT, WARN_LOW_CONTRAST, WARN_BLURRY

Note: samples/clean_page.png is a computer-rendered test page; sim_* files are
SIMULATED damage. Replace with real public-domain scans for the demo.
