import os
import shutil
from pathlib import Path

import pytesseract


def configure_tesseract():
    # Custom executable location
    custom_path = os.environ.get("TESSERACT_CMD", "").strip()

    if custom_path:
        executable = shutil.which(custom_path)

        if executable is None and Path(custom_path).is_file():
            executable = custom_path

    else:
        # Works on Windows/Linux when Tesseract is in PATH
        executable = shutil.which("tesseract")

        # Windows installation fallback
        if executable is None and os.name == "nt":
            program_files = os.environ.get("ProgramFiles")

            if program_files:
                candidate = (
                    Path(program_files)
                    / "Tesseract-OCR"
                    / "tesseract.exe"
                )

                if candidate.is_file():
                    executable = str(candidate)

    if not executable:
        raise RuntimeError(
            "Tesseract OCR not found. Install Tesseract "
            "and Tamil language data, then configure PATH "
            "or TESSERACT_CMD."
        )

    pytesseract.pytesseract.tesseract_cmd = str(executable)

    return str(executable)