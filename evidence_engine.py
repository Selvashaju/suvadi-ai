import unicodedata
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz


@dataclass
class EvidenceResult:
    text: str
    confidence: float
    agreement: float
    score: float
    status: str
    alternatives: list[str]


def normalise(text: str) -> str:
    """Normalize OCR text before comparison."""

    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Tesseract may insert zero-width characters around Tamil marks.
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    return " ".join(text.split())


def calculate_evidence_score(
    confidence: float,
    agreement: float,
) -> float:
    """60% OCR confidence + 40% variant agreement."""

    confidence = max(0.0, min(100.0, confidence))
    agreement = max(0.0, min(100.0, agreement))

    return (0.60 * confidence) + (0.40 * agreement)


def classify_status(score: float) -> str:

    if score >= 85:
        return "HIGH"

    if score >= 60:
        return "REVIEW"

    return "UNREADABLE"


def boxes_are_near(
    token_a: dict,
    token_b: dict,
    tolerance: int = 40,
) -> bool:
    """Check whether two OCR tokens are approximately in the same place."""

    return (
        abs(token_a["x"] - token_b["x"]) <= tolerance
        and abs(token_a["y"] - token_b["y"]) <= tolerance
    )


def text_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """Return similarity between normalized OCR strings."""

    a = normalise(text_a)
    b = normalise(text_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 100.0

    return fuzz.ratio(a, b)


def tokens_match(
    token_a: dict,
    token_b: dict,
) -> bool:
    """
    Conservative token agreement.

    Exact normalized text is always accepted.

    Fuzzy matching is only accepted when:
    - tokens are spatially close
    - similarity is very high
    - string lengths are reasonably close
    """

    if not boxes_are_near(token_a, token_b):
        return False

    a = normalise(token_a["text"])
    b = normalise(token_b["text"])

    if not a or not b:
        return False

    # Strongest possible agreement.
    if a == b:
        return True

    similarity = fuzz.ratio(a, b)

    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))

    if longer == 0:
        return False

    length_ratio = shorter / longer

    # Conservative fuzzy agreement.
    return (
        similarity >= 92
        and length_ratio >= 0.85
    )


def find_nearby_token(
    reference_token: dict,
    candidate_tokens: list[dict],
) -> Optional[dict]:
    """Find the closest token at approximately the same position."""

    candidates = []

    for candidate in candidate_tokens:

        if not boxes_are_near(
            reference_token,
            candidate,
        ):
            continue

        similarity = text_similarity(
            reference_token["text"],
            candidate["text"],
        )

        distance = (
            abs(reference_token["x"] - candidate["x"])
            + abs(reference_token["y"] - candidate["y"])
        )

        candidates.append(
            (
                similarity,
                -distance,
                candidate,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    return candidates[0][2]


def compare_variants(
    variants: dict[str, list[dict]],
) -> list[EvidenceResult]:
    """
    Compare OCR results from the available preprocessing variants.

    The first available variant provides positional reference tokens.

    Agreement is counted only when the candidate passes the conservative
    tokens_match() check.

    Conflicting readings remain visible in alternatives.
    """

    available = [
        tokens
        for tokens in variants.values()
        if tokens
    ]

    if not available:
        return []

    reference_tokens = available[0]
    results = []

    for reference in reference_tokens:

        supporting_tokens = [reference]
        nearby_tokens = [reference]

        for candidate_tokens in available[1:]:

            candidate = find_nearby_token(
                reference,
                candidate_tokens,
            )

            if candidate is None:
                continue

            nearby_tokens.append(candidate)

            if tokens_match(
                reference,
                candidate,
            ):
                supporting_tokens.append(candidate)

        agreement = (
            len(supporting_tokens)
            / len(available)
        ) * 100

        confidence = sum(
            token["confidence"]
            for token in supporting_tokens
        ) / len(supporting_tokens)

        score = calculate_evidence_score(
            confidence,
            agreement,
        )

        status = classify_status(score)

        alternatives = []

        for token in nearby_tokens:

            text = normalise(token["text"])

            if text and text not in alternatives:
                alternatives.append(text)

        results.append(
            EvidenceResult(
                text=normalise(reference["text"]),
                confidence=round(
                    confidence,
                    2,
                ),
                agreement=round(
                    agreement,
                    2,
                ),
                score=round(
                    score,
                    2,
                ),
                status=status,
                alternatives=alternatives,
            )
        )

    return results


if __name__ == "__main__":

    print("Evidence Engine - Conservative Voting Test")
    print("=" * 60)

    variants = {
        "gray": [
            {
                "text": "உம்மை",
                "confidence": 90,
                "x": 100,
                "y": 100,
                "width": 100,
                "height": 30,
            }
        ],
        "threshold": [
            {
                "text": "உம்மை",
                "confidence": 88,
                "x": 102,
                "y": 101,
                "width": 100,
                "height": 30,
            }
        ],
        "sharpened": [
            {
                "text": "வேறு",
                "confidence": 85,
                "x": 101,
                "y": 100,
                "width": 100,
                "height": 30,
            }
        ],
    }

    results = compare_variants(variants)

    for result in results:
        print(result)
