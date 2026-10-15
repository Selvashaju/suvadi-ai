from pathlib import Path
import json
import sys

from historical_preprocessor import preprocess_historical_image
from ocr_engine import run_ocr
from evidence_engine import compare_variants, normalise


def find_matching_tokens(result_text, ocr_results):
    """
    Find OCR tokens that correspond to the final evidence result.
    Used only for attaching provenance information to the JSON output.
    """

    matches = []

    normalized_result = normalise(result_text)

    for variant_name, tokens in ocr_results.items():
        for token in tokens:

            token_text = normalise(token.get("text", ""))

            if token_text == normalized_result:
                matches.append(
                    {
                        "variant": variant_name,
                        "token": token,
                    }
                )

    return matches


def build_evidence_record(
    token_id,
    result,
    ocr_results,
):
    """
    Convert an EvidenceResult into the frontend/API JSON contract.
    """

    matches = find_matching_tokens(
        result.text,
        ocr_results,
    )

    # Use the first matching OCR token as the reference position.
    bounding_box = None

    if matches:
        reference_token = matches[0]["token"]

        bounding_box = {
            "x": reference_token.get("x"),
            "y": reference_token.get("y"),
            "width": reference_token.get("width"),
            "height": reference_token.get("height"),
        }

    source_variants = [
        match["variant"]
        for match in matches
    ]

    # Remove duplicates while preserving order.
    source_variants = list(
        dict.fromkeys(source_variants)
    )

    source_variant = (
        source_variants[0]
        if source_variants
        else None
    )

    return {
        "token_id": token_id,
        "text": result.text,
        "confidence": result.confidence,
        "agreement": result.agreement,
        "score": result.score,
        "status": result.status,
        "evidence_state": result.status,
        "bounding_box": bounding_box,
        "source_variant": source_variant,
        "source_variants": source_variants,
        "alternatives": result.alternatives,
    }


def main():

    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <image>")
        raise SystemExit(1)

    image_path = sys.argv[1]

    print("SUVADI AI - Historical Tamil OCR")
    print("=" * 55)
    print(f"Input: {image_path}")

    # ---------------------------------------------------------
    # 1. Historical preprocessing
    # ---------------------------------------------------------

    print("\n[1/3] Preprocessing...")

    variants = preprocess_historical_image(
        image_path
    )

    for name, path in variants.items():
        print(
            f"  {name:12} -> {path}"
        )

    # ---------------------------------------------------------
    # 2. Tamil OCR
    # ---------------------------------------------------------

    print("\n[2/3] Tamil OCR...")

    ocr_results = {}

    for name, path in variants.items():

        print(
            f"  Processing {name}..."
        )

        tokens = run_ocr(
            path,
            lang="tam",
        )

        ocr_results[name] = tokens

        print(
            f"    {len(tokens)} OCR tokens"
        )

    # ---------------------------------------------------------
    # 3. Evidence comparison
    # ---------------------------------------------------------

    print("\n[3/3] Evidence comparison...")

    evidence_results = compare_variants(
        ocr_results
    )

    print("\nEvidence Results")
    print("=" * 55)

    for number, result in enumerate(
        evidence_results,
        start=1,
    ):

        print(
            f"{number}. {result.text} | "
            f"Confidence: {result.confidence:.1f}% | "
            f"Agreement: {result.agreement:.1f}% | "
            f"Score: {result.score:.1f} | "
            f"Status: {result.status}"
        )

    # ---------------------------------------------------------
    # 4. Build frontend-ready evidence records
    # ---------------------------------------------------------

    evidence_records = []

    for token_id, result in enumerate(
        evidence_results,
        start=1,
    ):

        record = build_evidence_record(
            token_id,
            result,
            ocr_results,
        )

        evidence_records.append(record)

    # ---------------------------------------------------------
    # 5. Save structured JSON
    # ---------------------------------------------------------

    output_dir = Path("output")

    output_dir.mkdir(
        exist_ok=True
    )

    json_path = (
        output_dir
        / "evidence_results.json"
    )

    report = {
        "project": "SUVADI AI",
        "input_image": str(
            Path(image_path)
        ),
        "language": "tam",

        "preprocessing_variants": {
            name: str(path)
            for name, path
            in variants.items()
        },

        "ocr_token_counts": {
            name: len(tokens)
            for name, tokens
            in ocr_results.items()
        },

        "evidence_results":
            evidence_records,
    }

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------

    high_count = sum(
        1
        for result in evidence_results
        if result.status == "HIGH"
    )

    review_count = sum(
        1
        for result in evidence_results
        if result.status == "REVIEW"
    )

    unreadable_count = sum(
        1
        for result in evidence_results
        if result.status == "UNREADABLE"
    )

    print("\nEvidence Summary")
    print("=" * 55)
    print(f"HIGH        : {high_count}")
    print(f"REVIEW      : {review_count}")
    print(f"UNREADABLE  : {unreadable_count}")

    print("\nStructured report saved:")
    print(f"  {json_path}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
