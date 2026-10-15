"""Local SQLite draft for Member 3 review persistence; coordinate with Member 4.

Records both the latest human decision and an append-only decision-event log.
Never overwrites already reviewed OCR tokens during a rerun of the UI.
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
    connection = sqlite3.connect(DB_PATH, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("""CREATE TABLE IF NOT EXISTS pages (
        page_key TEXT PRIMARY KEY, book_title TEXT NOT NULL,
        author TEXT NOT NULL, publication_year INTEGER,
        page_number INTEGER NOT NULL, image_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS tokens (
        page_key TEXT NOT NULL, token_id TEXT NOT NULL,
        token_order INTEGER NOT NULL, token_json TEXT NOT NULL,
        PRIMARY KEY(page_key, token_id),
        FOREIGN KEY(page_key) REFERENCES pages(page_key)
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS decision_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_key TEXT NOT NULL, token_id TEXT NOT NULL,
        reviewer_action TEXT NOT NULL, original_value TEXT NOT NULL,
        corrected_value TEXT, timestamp TEXT NOT NULL
    )""")
    connection.commit()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_page(image_bytes: bytes, book_title: str, author: str, year: int, number: int) -> str:
    # Stable identity: same image+book metadata resumes the same review.
    identifier = json.dumps([book_title, author, year, number], ensure_ascii=False)
    page_key = hashlib.sha256(image_bytes + identifier.encode("utf-8")).hexdigest()
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = PAGE_DIR / f"{page_key}.png"
    if not image_path.is_file():
        with Image.open(io.BytesIO(image_bytes)) as original:
            original.convert("RGB").save(image_path, format="PNG")
    with _connect() as conn:
        conn.execute("""INSERT OR IGNORE INTO pages
            (page_key, book_title, author, publication_year, page_number, image_path, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (page_key, book_title, author, int(year), int(number), str(image_path), _utc()))
    return page_key


def list_pages() -> list[dict]:
    with _connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM pages ORDER BY created_at DESC, rowid DESC")]


def load_page(page_key: str) -> tuple[dict, bytes]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM pages WHERE page_key=?", (page_key,)).fetchone()
    if row is None:
        raise KeyError("Page not found")
    page = dict(row)
    # Image filename is derived from the requested page key, not arbitrary database path.
    return page, (PAGE_DIR / f"{page_key}.png").read_bytes()


def get_tokens(page_key: str | None) -> list[dict]:
    if not page_key:
        return []
    with _connect() as conn:
        return [json.loads(row[0]) for row in conn.execute(
            "SELECT token_json FROM tokens WHERE page_key=? ORDER BY token_order", (page_key,))]


def add_tokens_if_absent(page_key: str, tokens: list[dict]) -> bool:
    if not tokens:
        return False
    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM tokens WHERE page_key=? LIMIT 1", (page_key,)).fetchone()
        if exists:
            return False  # preserves previously reviewed words and audit history
        for i, token in enumerate(tokens):
            conn.execute("""INSERT INTO tokens(page_key, token_id, token_order, token_json)
                VALUES(?,?,?,?)""", (page_key, token["token_id"], i,
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
        row = conn.execute("SELECT token_json FROM tokens WHERE page_key=? AND token_id=?",
                           (page_key, token_id)).fetchone()
        if row is None:
            raise KeyError("OCR token not found")
        token = json.loads(row[0])
        if action == "edit" and not (correction or "").strip():
            raise ValueError("A corrected Tamil word must be provided")
        final_value = (token["raw_value"] if action == "accept" else
                       correction.strip() if action == "edit" else None)
        token.update({"status": status_map[action],
                      "review_action": action if action != "reset" else None,
                      "final_value": final_value,
                      "reviewed_at": _utc() if action != "reset" else None})
        conn.execute("UPDATE tokens SET token_json=? WHERE page_key=? AND token_id=?",
                     (json.dumps(token, ensure_ascii=False), page_key, token_id))
        conn.execute("""INSERT INTO decision_events
            (page_key,token_id,reviewer_action,original_value,corrected_value,timestamp)
            VALUES(?,?,?,?,?,?)""", (page_key, token_id, action, token["original_value"],
                                      final_value, _utc()))


def reset_all(page_key: str):
    for token in get_tokens(page_key):
        save_decision(page_key, token["token_id"], "reset")


def get_audit_events(page_key: str) -> list[dict]:
    with _connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM decision_events WHERE page_key=? ORDER BY event_id", (page_key,))]
