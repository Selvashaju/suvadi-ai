"""Human-verified page archive, keyed by the persistent page identity.

Legacy records are readable. New saves use page_key as their identity so that
re-saving the same page updates the existing entry rather than adding a copy.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ARCHIVE_PATH = Path(__file__).resolve().parent / "data" / "archive.json"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_archive() -> list[dict]:
    if not ARCHIVE_PATH.is_file():
        return []
    with ARCHIVE_PATH.open("r", encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError("Archive file must contain a JSON list")
    return records


def save_archive_record(
    book_title: str,
    author: str,
    page_number: int,
    tokens: list[dict],
    *,
    page_key: str,
    source_sha256: str | None,
) -> dict:
    """Insert/update a reviewed page; never append a duplicate page_key.

    source_sha256 may be None for an old SQLite row created before this field
    existed. Never substitute the saved/converted PNG hash for a missing hash
    of the originally uploaded bytes.
    """
    if not isinstance(page_key, str) or not page_key.strip():
        raise ValueError("A saved page_key is required for archive persistence")
    if source_sha256 is not None and (
        len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256.lower())
    ):
        raise ValueError("source_sha256 must be a SHA-256 hex digest or None")

    if any(t.get("status") in ("review", "needs_review") for t in tokens):
        raise ValueError("Finish all pending human reviews before archiving")
    approved = [
        token for token in tokens
        if token.get("status") == "verified"
        and token.get("review_action") in ("accept", "edit")
        and token.get("final_value")
    ]
    if not approved:
        raise ValueError("At least one human-verified word is required")

    records = load_archive()
    matching = next(
        (record for record in records if record.get("page_key") == page_key),
        None,
    )
    now = _utc()
    record = {
        "record_id": matching.get("record_id") if matching else str(uuid4()),
        "created_at": matching.get("created_at", now) if matching else now,
        "updated_at": now,
        "sample_data": False,
        "page_key": page_key,
        "source_sha256": source_sha256,
        "book_title": book_title,
        "author": author,
        "page_number": int(page_number),
        "verified_text": " ".join(t["final_value"] for t in approved),
        "reviewed_tokens": tokens,
    }

    # Collapse any existing duplicated records for this page. Unrelated and
    # older records without page_key are left untouched, not guessed/migrated.
    retained = [r for r in records if r.get("page_key") != page_key]
    retained.append(record)

    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE_PATH.with_name(ARCHIVE_PATH.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(retained, destination, ensure_ascii=False, indent=2)
        os.replace(temporary, ARCHIVE_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def search_archive(query: str) -> list[dict]:
    records = load_archive()
    if not query.strip():
        return records
    term = query.strip().casefold()
    return [
        record for record in records
        if term in record.get("book_title", "").casefold()
        or term in record.get("author", "").casefold()
        or term in record.get("verified_text", "").casefold()
    ]
