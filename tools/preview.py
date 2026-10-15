"""
One-command visual check:

    python tools/preview.py samples/sim_faded.png

Saves out/<name>_preview.png (original | gray | threshold | sharpened)
and opens it (macOS `open`).
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import preprocess_image  # noqa: E402


def tile(images, height=900):
    row = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        s = height / im.shape[0]
        row.append(cv2.resize(im, (int(im.shape[1] * s), height)))
        row.append(np.full((height, 12, 3), 255, np.uint8))
    return np.hstack(row[:-1])


def main(path):
    r = preprocess_image(path, page_id=Path(path).stem)
    meta = {k: v for k, v in r["metadata"].items()
            if k not in ("rotation_matrix", "perspective_matrix", "settings")}
    print(json.dumps({"status": r["status"], "error_code": r["error_code"], **meta},
                     indent=2, ensure_ascii=False))
    if r["status"] != "ok":
        return
    v = r["variants"]
    sheet = tile([r["original_rgb"], v["gray"], v["threshold"], v["sharpened"]])
    out = Path("out")
    out.mkdir(exist_ok=True)
    dst = out / f"{Path(path).stem}_preview.png"
    cv2.imwrite(str(dst), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("saved", dst)
    if platform.system() == "Darwin":
        subprocess.run(["open", str(dst)])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/clean_page.png")
