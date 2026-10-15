
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

ARCHIVE_PATH = Path(__file__).resolve().parent / "data" / "archive.json"


def load_archive():
    if not ARCHIVE_PATH.exists():
        return []

    with ARCHIVE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_archive_record(book_title, author, page_number, tokens):
    approved = [
        token for token in tokens
        if token.get("status") == "verified"
        and token.get("review_action") in ("accept", "edit")
        and token.get("final_value")
    ]

    record = {
        "record_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_data": True,
        "book_title": book_title,
        "author": author,
        "page_number": page_number,
        "verified_text": " ".join(
            token["final_value"] for token in approved
        ),
        "reviewed_tokens": tokens
    }

    records = load_archive()
    records.append(record)

    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ARCHIVE_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return record


def search_archive(query):
    records = load_archive()

    if not query.strip():
        return records

    query = query.strip().casefold()

    return [
        record for record in records
        if query in record.get("book_title", "").casefold()
        or query in record.get("author", "").casefold()
        or query in record.get("verified_text", "").casefold()
    ]
