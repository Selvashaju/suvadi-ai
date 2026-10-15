"""SUVADI AI: read-only Tamil OCR and modern-font preview for Streamlit.

Call render_modern_reader(preprocess_result) within the restoration page.
This module does NOT approve, alter, or export any historical verification record.
Member 2's OCR engine should ultimately replace _recognize_variants().
"""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import os
import unicodedata
from pathlib import Path

import pytesseract
import streamlit as st

_ROOT = Path(__file__).resolve().parent
_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

if os.name == "nt" and _WINDOWS_TESSERACT.is_file():
    pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)

_VARIANT_LABELS = {
    "gray": "சாம்பல் நிறப் படம்",
    "threshold": "கருப்பு வெள்ளைப் படம்",
    "sharpened": "எழுத்துத் தெளிவாக்கப்பட்ட படம்",
}


def readable_unicode(text: str) -> str:
    """Create a readable preview; preserve the raw OCR text separately."""
    return unicodedata.normalize("NFC", text).replace("\u200c", "").replace("\u200d", "")


def _font_css() -> str:
    font = _ROOT / "assets" / "fonts" / "NotoSansTamil-Regular.ttf"
    if font.is_file():
        encoded = base64.b64encode(font.read_bytes()).decode("ascii")
        font_rule = (
            "@font-face{font-family:'SUVADI-Tamil';"
            f"src:url(data:font/ttf;base64,{encoded}) format('truetype');"
            "font-weight:normal;font-style:normal;}"
        )
    else:
        font_rule = ""
    return (
        "<style>" + font_rule +
        ".suvadi-readable{font-family:'SUVADI-Tamil','Nirmala UI',sans-serif;"
        "white-space:normal;overflow-wrap:anywhere;line-height:2.05;"
        "padding:18px;border:1px solid #9996;border-radius:12px;}"
        "</style>"
    )


def _recognize_variants(result: dict) -> dict[str, str]:
    # Temporary standalone OCR preview. Member 2 owns the final OCR engine.
    languages = pytesseract.get_languages(config="")
    if "tam" not in languages:
        raise RuntimeError("Tamil language data not installed")
    return {
        name: pytesseract.image_to_string(
            image, lang="tam", config="--psm 6"
        )
        for name, image in result["variants"].items()
        if name in _VARIANT_LABELS
    }


def render_modern_reader(result: dict) -> None:
    """Add automatic Tamil OCR and modern-font reading below the restoration views."""
    if not result or result.get("status") != "ok":
        return

    st.divider()
    st.subheader("📖 தெளிவான தமிழ் வாசிப்பு")
    st.info(
        "இந்தப் பகுதி சீரமைக்கப்பட்ட படங்களில் உள்ள எழுத்துகளை வாசித்து "
        "புதிய எழுத்துருவில் காட்டுகிறது. இது மனிதர் சரிபார்த்த இறுதி உரை அல்ல."
    )

    metadata = result.get("metadata", {})
    identity = hashlib.sha256(
        (str(metadata.get("source_sha256", "")) +
         repr(metadata.get("settings", {}))).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("_tamil_reader_identity") != identity:
        st.session_state["_tamil_reader_identity"] = identity
        st.session_state.pop("_tamil_reader_ocr", None)

    if "_tamil_reader_ocr" not in st.session_state:
        try:
            with st.spinner("மூன்று படங்களிலும் தமிழ் எழுத்துகள் கண்டறியப்படுகின்றன..."):
                st.session_state["_tamil_reader_ocr"] = _recognize_variants(result)
        except Exception:
            logging.exception("Tamil reading preview OCR failed")
            st.error(
                "தமிழ் எழுத்தறிதல் இயங்கவில்லை. கணினியில் டெசராக்ட் நிரலும் "
                "தமிழ் மொழித் தரவும் நிறுவப்பட்டுள்ளனவா எனச் சரிபார்க்கவும்."
            )
            return

    if st.button("🔄 எழுத்தறிதலை மீண்டும் இயக்கவும்", key="tamil_reader_retry"):
        st.session_state.pop("_tamil_reader_ocr", None)
        st.rerun()

    outputs = st.session_state["_tamil_reader_ocr"]
    keys = list(outputs.keys())
    if not keys:
        st.warning("வாசிப்பதற்கான படங்கள் கிடைக்கவில்லை.")
        return

    chosen = st.selectbox(
        "எந்தப் படத்தின் எழுத்தறிதலைப் பார்க்க வேண்டும்?",
        keys,
        format_func=lambda k: _VARIANT_LABELS.get(k, k),
        key="tamil_reader_variant",
    )
    raw = outputs[chosen]
    if not raw.strip():
        st.warning("இந்தப் படத்தில் வாசிக்கக்கூடிய தமிழ் உரை கிடைக்கவில்லை.")
        return

    st.text_area(
        "🧾 கணினி கண்டறிந்த அசல் உரை (திருத்தப்படாதது)",
        value=raw,
        height=200,
        disabled=True,
    )
    normalized = readable_unicode(raw)
    text_size = st.slider(
        "🔎 தமிழ் எழுத்தின் அளவு", min_value=18, max_value=38,
        value=25, key="tamil_reader_font_size"
    )
    st.markdown(_font_css(), unsafe_allow_html=True)
    escaped = html.escape(normalized).replace("\n", "<br>")
    st.markdown(
        f'<div class="suvadi-readable" style="font-size:{text_size}px">'
        f'{escaped}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "குறிப்பு: இது அதே எழுத்தறிதல் உரையின் புதிய எழுத்துரு வடிவம். "
        "சொற்களின் பொருளும் எழுத்தறிதல் பிழைகளும் தானாகச் சரிசெய்யப்படவில்லை."
    )
    st.text_area(
        "✍️ இன்றைய தமிழில் பொருள் விளக்க வரைவு (மனிதர் எழுதலாம்)",
        key=f"tamil_reader_draft_{identity}",
        placeholder="மூல உரையுடன் ஒப்பிட்டு, பொருள் விளக்கத்தை இங்கே எழுதவும்...",
        height=125,
    )
    st.caption(
        "இந்த வரைவு தனியாக வைக்கப்படுகிறது. ஆதாரச் சரிபார்ப்பு இல்லாமல் "
        "இது சரிபார்க்கப்பட்ட உரையாகச் சேமிக்கப்படாது."
    )
