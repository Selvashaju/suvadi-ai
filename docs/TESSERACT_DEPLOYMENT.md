# SUVADI AI — Windows and Linux Tesseract Deployment

These are **environment prerequisites** in addition to the Python packages from `requirements.txt`.
Tesseract is a separate executable; `pip install pytesseract` does **not** install it.

## Windows

1. Install Tesseract OCR from a trusted Windows installer, selecting the **Tamil (`tam`)** language during setup, or put `tam.traineddata` into Tesseract's `tessdata` folder.
2. If Tesseract is added to `PATH`, `tesseract --version` should work in a fresh PowerShell window.
3. Otherwise, the included `tesseract_setup.py` checks the usual `ProgramFiles\\Tesseract-OCR\\tesseract.exe` location.
4. If it is installed elsewhere, specify the full path **for your own machine**:

   ```powershell
   $env:TESSERACT_CMD = 'D:\\Apps\\Tesseract-OCR\\tesseract.exe'
   ```

5. Check the language data (use the full executable path if it isn't on PATH):

   ```powershell
   tesseract --list-langs
   ```

   The result must include `tam`.

## Ubuntu / Debian Linux

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-tam
which tesseract
tesseract --version
tesseract --list-langs
```

Ensure `tam` appears in the language list. For a nonstandard installation:

```bash
export TESSERACT_CMD=/opt/tesseract/bin/tesseract
```

## Python setup (Windows or Linux)

From the repository root in an activated Python environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_member3_hotfix.py
python -m streamlit run SUVADI_AI_TAMIL_UI.py
```

Test the six Streamlit screens, upload an image, run preprocessing/OCR, review a
word, archive the page, and archive the same page again. It should update the
same record. **Do not claim real OCR accuracy without ground-truth evaluation.**

## Data integrity / migration notes

- `data/` holds actual local SQLite databases, original-page representations and
  archive JSON. Keep this directory out of Git (`.gitignore` already excludes it).
- The PR #2 SQLite migration uses `ALTER TABLE` to add `source_sha256` without
  replacing the database or deleting reviews/audit history.
- `page_key` remains the existing PR #1 stable identity for an upload plus book
  metadata; the archive uses it to update records rather than append duplicates.
- SHA-256 hashes **original uploaded file bytes**, before PNG conversion.
- For pages saved before PR #2, a source hash may be `null` until the **exact
  original upload** is submitted again. A hash of the stored converted PNG is
  not silently misrepresented as the original source hash.
- Archive records created before PR #2 without `page_key` remain readable; their
  identity cannot be inferred reliably, so they are not automatically deduplicated.
- The JSON archive uses an atomic file replacement for completed saves, but
  multiple independent writers/processes still require coordinated locking or
  Member 4's transactional storage design before production multi-user use.

## PR #2 validation checklist

- [ ] Automated tests pass.
- [ ] App starts on Windows and can find the installed `tam` language.
- [ ] Archive a human-reviewed page twice; one entry remains.
- [ ] SQLite `pages` table contains `page_key` and `source_sha256`.
- [ ] The two duplicated Tamil prompts are fixed.
- [ ] On Linux with `tesseract-ocr-tam`, run an OCR smoke test.
- [ ] Commit and push **only** `hotfix/member3-final-corrections`, then open PR #2 against `main`.
