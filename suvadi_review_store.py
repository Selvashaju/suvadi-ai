"""Local SQLite review persistence for SUVADI AI Member 3.

- Idempotent page uploads preserve reviewed OCR tokens and audit history.
- New uploads retain SHA-256 of the exact original image bytes.
- Existing Member 3 databases gain a nullable provenance column in place.
  Old source hashes are unknown unless the original upload is submitted again.
- Member 4 should review the final storage integration contract.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent / "data"
DB_PATH = BASE / "member3_reviews.sqlite3"
PAGE_DIR = BASE / "pages"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect():
    BASE.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""CREATE TABLE IF NOT EXISTS pages (
            page_key TEXT PRIMARY KEY,
            book_title TEXT NOT NULL,
            author TEXT NOT NULL,
            publication_year INTEGER,
            page_number INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_sha256 TEXT
        )""")
        # Safe, non-destructive migration for databases created before PR #2.
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(pages)")
        }
        if "source_sha256" not in columns:
            conn.execute("ALTER TABLE pages ADD COLUMN source_sha256 TEXT")

        conn.execute("""CREATE TABLE IF NOT EXISTS tokens (
            page_key TEXT NOT NULL,
            token_id TEXT NOT NULL,
            token_order INTEGER NOT NULL,
            token_json TEXT NOT NULL,
            PRIMARY KEY(page_key, token_id),
            FOREIGN KEY(page_key) REFERENCES pages(page_key)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS decision_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_key TEXT NOT NULL,
            token_id TEXT NOT NULL,
            reviewer_action TEXT NOT NULL,
            original_value TEXT NOT NULL,
            corrected_value TEXT,
            timestamp TEXT NOT NULL
        )""")
        conn.commit()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()


def save_page(image_bytes: bytes, book_title: str, author: str, year: int, number: int) -> str:
    """Persist the original source hash, and return the existing stable page_key."""
    if not image_bytes:
        raise ValueError("An uploaded image is required")
    source_sha256 = hashlib.sha256(image_bytes).hexdigest()
    identifier = json.dumps(
        [book_title, author, year, number], ensure_ascii=False
    )
    # Preserve the PR #1 identity algorithm to keep old review data reachable.
    page_key = hashlib.sha256(
        image_bytes + identifier.encode("utf-8")
    ).hexdigest()
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = PAGE_DIR / f"{page_key}.png"
    if not image_path.is_file():
        with Image.open(io.BytesIO(image_bytes)) as original:
            original.convert("RGB").save(image_path, format="PNG")

    with _connect() as conn:
        conn.execute("""INSERT OR IGNORE INTO pages
            (page_key, book_title, author, publication_year,
             page_number, image_path, created_at, source_sha256)
            VALUES (?,?,?,?,?,?,?,?)""",
            (page_key, book_title, author, int(year), int(number),
             str(image_path), _utc(), source_sha256))
        # Backfill only if the exact same ORIGINAL upload bytes are available.
        conn.execute("""UPDATE pages SET source_sha256=?
            WHERE page_key=? AND source_sha256 IS NULL""",
            (source_sha256, page_key))
    return page_key


def list_pages() -> list[dict]:
    with _connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM pages ORDER BY created_at DESC, rowid DESC")]


def get_page_info(page_key: str | None) -> dict | None:
    if not page_key:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pages WHERE page_key=?", (page_key,)
        ).fetchone()
        return dict(row) if row else None


def load_page(page_key: str) -> tuple[dict, bytes]:
    page = get_page_info(page_key)
    if page is None:
        raise KeyError("Page not found")
    # Derive the path from page_key, not an arbitrary path stored in SQLite.
    return page, (PAGE_DIR / f"{page_key}.png").read_bytes()


def get_tokens(page_key: str | None) -> list[dict]:
    if not page_key:
        return []
    with _connect() as conn:
        return [json.loads(row[0]) for row in conn.execute(
            "SELECT token_json FROM tokens WHERE page_key=? ORDER BY token_order",
            (page_key,))]


def add_tokens_if_absent(page_key: str, tokens: list[dict]) -> bool:
    if not tokens:
        return False
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM tokens WHERE page_key=? LIMIT 1", (page_key,)
        ).fetchone()
        if exists:
            return False  # Do not overwrite earlier human review decisions.
        for i, token in enumerate(tokens):
            conn.execute("""INSERT INTO tokens
                (page_key, token_id, token_order, token_json)
                VALUES(?,?,?,?)""",
                (page_key, token["token_id"], i,
                 json.dumps(token, ensure_ascii=False)))
    return True


def save_decision(page_key: str, token_id: str, action: str, correction: str | None = None):
    status_map = {
        "accept": "verified", "edit": "verified", "reject": "rejected",
        "unreadable": "unreadable", "flag": "needs_review", "reset": "review",
    }
    if action not in status_map:
        raise ValueError("Invalid reviewer action")
    with _connect() as conn:
        row = conn.execute(
            "SELECT token_json FROM tokens WHERE page_key=? AND token_id=?",
            (page_key, token_id)
        ).fetchone()
        if row is None:
            raise KeyError("OCR token not found")
        token = json.loads(row[0])
        if action == "edit" and not (correction or "").strip():
            raise ValueError("A corrected Tamil word must be provided")
        final_value = (
            token["raw_value"] if action == "accept"
            else correction.strip() if action == "edit"
            else None
        )
        token.update({
            "status": status_map[action],
            "review_action": action if action != "reset" else None,
            "final_value": final_value,
            "reviewed_at": _utc() if action != "reset" else None,
        })
        conn.execute(
            "UPDATE tokens SET token_json=? WHERE page_key=? AND token_id=?",
            (json.dumps(token, ensure_ascii=False), page_key, token_id))
        conn.execute("""INSERT INTO decision_events
            (page_key, token_id, reviewer_action,
             original_value, corrected_value, timestamp)
            VALUES(?,?,?,?,?,?)""",
            (page_key, token_id, action, token["original_value"],
             final_value, _utc()))


def reset_all(page_key: str):
    for token in get_tokens(page_key):
        save_decision(page_key, token["token_id"], "reset")


def get_audit_events(page_key: str) -> list[dict]:
    with _connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM decision_events WHERE page_key=? ORDER BY event_id",
            (page_key,))]
