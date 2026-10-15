"""Run: python -m pytest -q tests/test_member3_hotfix.py

Uses synthetic in-memory images and temporary files only. It never edits the
real SUVADI AI data/ folder and does not require a running Tesseract process.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import archive
import suvadi_review_store as store


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "BASE", tmp_path / "data")
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "data" / "member3_reviews.sqlite3")
    monkeypatch.setattr(store, "PAGE_DIR", tmp_path / "data" / "pages")
    monkeypatch.setattr(archive, "ARCHIVE_PATH", tmp_path / "data" / "archive.json")
    return tmp_path


@pytest.fixture
def image_bytes():
    out = io.BytesIO()
    Image.new("RGB", (48, 48), (255, 255, 230)).save(out, format="PNG")
    return out.getvalue()


def reviewed(value="தமிழ்"):
    return {
        "token_id": "token-01",
        "raw_value": "தமிழ்",
        "original_value": "தமிழ்",
        "status": "verified",
        "review_action": "edit" if value != "தமிழ்" else "accept",
        "final_value": value,
    }


def test_page_identity_and_original_image_sha(isolated_data, image_bytes):
    first = store.save_page(image_bytes, "திருக்குறள்", "திருவள்ளுவர்", 1900, 1)
    second = store.save_page(image_bytes, "திருக்குறள்", "திருவள்ளுவர்", 1900, 1)
    assert first == second
    page = store.get_page_info(first)
    assert page["page_key"] == first
    assert page["source_sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert len(store.list_pages()) == 1
    loaded, restored_bytes = store.load_page(first)
    assert loaded["page_key"] == first
    assert restored_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_different_pages_get_different_keys(isolated_data, image_bytes):
    first = store.save_page(image_bytes, "திருக்குறள்", "திருவள்ளுவர்", 1900, 1)
    second = store.save_page(image_bytes, "திருக்குறள்", "திருவள்ளுவர்", 1900, 2)
    assert first != second
    assert store.get_page_info(first)["source_sha256"] == store.get_page_info(second)["source_sha256"]


def test_sqlite_migration_preserves_reviews_and_audit(isolated_data, image_bytes):
    store.BASE.mkdir(parents=True)
    with sqlite3.connect(store.DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE pages (
            page_key TEXT PRIMARY KEY, book_title TEXT NOT NULL,
            author TEXT NOT NULL, publication_year INTEGER,
            page_number INTEGER NOT NULL, image_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE tokens (
            page_key TEXT NOT NULL, token_id TEXT NOT NULL,
            token_order INTEGER NOT NULL, token_json TEXT NOT NULL,
            PRIMARY KEY(page_key,token_id)
        );
        CREATE TABLE decision_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_key TEXT NOT NULL, token_id TEXT NOT NULL,
            reviewer_action TEXT NOT NULL, original_value TEXT NOT NULL,
            corrected_value TEXT, timestamp TEXT NOT NULL
        );
        """)
        book_title, author, year, number = "திருக்குறள்", "திருவள்ளுவர்", 1900, 1
        identity = json.dumps([book_title, author, year, number], ensure_ascii=False)
        key = hashlib.sha256(image_bytes + identity.encode("utf-8")).hexdigest()
        conn.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?)",
                     (key, book_title, author, year, number, "old.png", "earlier"))
        conn.execute("INSERT INTO tokens VALUES (?,?,?,?)",
                     (key, "token-01", 0, json.dumps(reviewed(), ensure_ascii=False)))
        conn.execute("INSERT INTO decision_events(page_key,token_id,reviewer_action,original_value,corrected_value,timestamp) VALUES (?,?,?,?,?,?)",
                     (key, "token-01", "accept", "தமிழ்", "தமிழ்", "earlier"))
    # Migration does NOT invent a digest from the converted/saved PNG.
    assert store.get_page_info(key)["source_sha256"] is None
    assert store.get_tokens(key)[0]["status"] == "verified"
    assert len(store.get_audit_events(key)) == 1
    # If the exact original upload is supplied, safely backfill provenance.
    assert store.save_page(image_bytes, book_title, author, year, number) == key
    assert store.get_page_info(key)["source_sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert len(store.get_audit_events(key)) == 1
    assert store.get_tokens(key)[0]["status"] == "verified"


def test_review_idempotence_and_decision_audit(isolated_data, image_bytes):
    key = store.save_page(image_bytes, "நூல்", "ஆசிரியர்", 1900, 1)
    original = dict(reviewed())
    original.update(status="review", review_action=None, final_value=None)
    assert store.add_tokens_if_absent(key, [original]) is True
    store.save_decision(key, "token-01", "edit", "பழந்தமிழ்")
    assert store.add_tokens_if_absent(key, [original]) is False
    assert store.get_tokens(key)[0]["final_value"] == "பழந்தமிழ்"
    assert len(store.get_audit_events(key)) == 1


def test_archive_insert_then_update_not_duplicate(isolated_data):
    digest = "a" * 64
    first = archive.save_archive_record(
        "நூல்", "ஆசிரியர்", 3, [reviewed()],
        page_key="page-3", source_sha256=digest)
    second = archive.save_archive_record(
        "நூல்", "ஆசிரியர்", 3, [reviewed("திருத்தம்")],
        page_key="page-3", source_sha256=digest)
    saved = archive.load_archive()
    assert len(saved) == 1
    assert first["record_id"] == second["record_id"]
    assert saved[0]["verified_text"] == "திருத்தம்"
    assert saved[0]["sample_data"] is False
    assert saved[0]["page_key"] == "page-3"
    assert saved[0]["source_sha256"] == digest


def test_archive_rejected_unreadable_excluded(isolated_data):
    tokens = [reviewed()]
    tokens += [
        {"status": "rejected", "review_action": "reject", "final_value": None},
        {"status": "unreadable", "review_action": "unreadable", "final_value": None},
    ]
    record = archive.save_archive_record(
        "நூல்", "ஆசிரியர்", 1, tokens, page_key="page", source_sha256="b" * 64)
    assert record["verified_text"] == "தமிழ்"


def test_archive_does_not_save_pending_or_unverified(isolated_data):
    with pytest.raises(ValueError, match="pending"):
        archive.save_archive_record(
            "நூல்", "ஆசிரியர்", 1,
            [reviewed(), {"status": "needs_review"}],
            page_key="page", source_sha256="b" * 64)
    with pytest.raises(ValueError, match="verified"):
        archive.save_archive_record(
            "நூல்", "ஆசிரியர்", 1, [{"status": "rejected"}],
            page_key="page", source_sha256="b" * 64)
    assert archive.load_archive() == []


def test_archive_legacy_records_are_preserved(isolated_data):
    archive.ARCHIVE_PATH.parent.mkdir(parents=True)
    archive.ARCHIVE_PATH.write_text(
        json.dumps([{"record_id": "old", "book_title": "பழைய நூல்",
                     "author": "பழைய ஆசிரியர்", "verified_text": "பழைய உரை"}], ensure_ascii=False),
        encoding="utf-8")
    archive.save_archive_record(
        "புதிய நூல்", "ஆசிரியர்", 2, [reviewed()],
        page_key="new-page", source_sha256="c" * 64)
    assert len(archive.load_archive()) == 2
    assert len(archive.search_archive("பழைய")) == 1
    assert len(archive.search_archive("தமிழ்")) == 1


def test_legacy_hash_is_honestly_unknown(isolated_data):
    record = archive.save_archive_record(
        "நூல்", "ஆசிரியர்", 1, [reviewed()],
        page_key="old-page", source_sha256=None)
    assert "source_sha256" in record and record["source_sha256"] is None


def test_duplicate_tamil_text_corrected():
    labels = json.loads((ROOT / "locale" / "ui_ta.json").read_text(encoding="utf-8"))
    assert "படச் சீரமைப்பு படச் சீரமைப்பு" not in labels["restoration_ui_023"]
    assert "எழுத்துணர்தல் எழுத்துணர்தல்" not in labels["restoration_ui_025"]
    assert all(labels[key] for key in ("archive_ui_legacy_hash", "archive_ui_save_failed"))


def test_archive_screen_passes_source_identity():
    source = (ROOT / "SUVADI_AI_TAMIL_UI.py").read_text(encoding="utf-8")
    assert 'page_key=page_info["page_key"]' in source
    assert "source_sha256=source_sha256" in source
    assert 'get_page_info(page_key)' in source


def test_archive_rejects_invalid_identity(isolated_data):
    with pytest.raises(ValueError, match="page_key"):
        archive.save_archive_record(
            "நூல்", "ஆசிரியர்", 1, [reviewed()],
            page_key="", source_sha256="a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        archive.save_archive_record(
            "நூல்", "ஆசிரியர்", 1, [reviewed()],
            page_key="page", source_sha256="not-a-digest")


def test_tesseract_config_uses_path_override(monkeypatch):
    import tesseract_setup
    import pytesseract
    monkeypatch.setenv("TESSERACT_CMD", "synthetic-tesseract")
    monkeypatch.setattr(tesseract_setup.shutil, "which",
                        lambda name: "/synthetic/bin/tesseract" if name == "synthetic-tesseract" else None)
    previous = pytesseract.pytesseract.tesseract_cmd
    try:
        assert tesseract_setup.configure_tesseract() == "/synthetic/bin/tesseract"
        assert pytesseract.pytesseract.tesseract_cmd == "/synthetic/bin/tesseract"
    finally:
        pytesseract.pytesseract.tesseract_cmd = previous


def test_tesseract_config_uses_system_path(monkeypatch):
    import tesseract_setup
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(tesseract_setup.shutil, "which",
                        lambda name: "/usr/bin/tesseract" if name == "tesseract" else None)
    assert tesseract_setup.configure_tesseract() == "/usr/bin/tesseract"
