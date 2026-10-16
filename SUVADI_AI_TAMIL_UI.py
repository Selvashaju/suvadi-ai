

import streamlit as st
import json
import logging
from pathlib import Path

# Load Tamil language translations
LOCALE_PATH = (
    Path(__file__).resolve().parent
    / "locale"
    / "ui_ta.json"
)

with LOCALE_PATH.open("r", encoding="utf-8") as file:
    UI_TEXT = json.load(file)


# Return the Tamil translation for a given key
def t(key):
    return UI_TEXT.get(key, f"மொழிபெயர்ப்பு இல்லை: {key}")


st.set_page_config(
    page_title=t("app_title"),
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tamil navigation

pages = [
    "🏠 " + t("upload"),
    "🖼️ " + t("restoration"),
    "🔍 " + t("review"),
    "✅ " + t("verified_text"),
    "📚 " + t("archive"),
    "📊 " + t("dashboard")
]



st.sidebar.title("📜 " + t("app_title"))
st.sidebar.write(t("app_subtitle"))

selected_page = st.sidebar.radio(
    t("navigation"),
    pages
)



st.sidebar.divider()
st.sidebar.caption(t("core_ui_001"))


# Page 1: Upload / restore an earlier page after application restart
if selected_page == pages[0]:
    from suvadi_review_store import save_page, list_pages, load_page

    st.title(t("upload_ui_001"))
    st.write(t("upload_ui_002"))
    c1, c2 = st.columns(2)
    with c1:
        book_title = st.text_input(t("book_title_label"), value=st.session_state.get("book_title", ""))
        author = st.text_input(t("author_label"), value=st.session_state.get("author", ""))
    with c2:
        publication_year = st.number_input(
            t("publication_year_label"), min_value=1000, max_value=2026,
            value=int(st.session_state.get("publication_year", 1900)))
        page_number = st.number_input(
            t("page_number_label"), min_value=1, value=int(st.session_state.get("page_number", 1)))

    uploaded_file = st.file_uploader(
        t("upload_ui_003"), type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption=t("upload_ui_006"), width="stretch")
        if st.button(t("upload_ui_004"), type="primary"):
            try:
                page_key = save_page(image_bytes, book_title, author,
                                     publication_year, page_number)
                st.session_state.update({
                    "active_page_key": page_key,
                    "uploaded_image": image_bytes,
                    "book_title": book_title,
                    "author": author,
                    "publication_year": publication_year,
                    "page_number": page_number,
                })
                st.session_state.pop("preprocess_result", None)
                st.session_state.pop("preprocess_source_hash", None)
                st.success(t("upload_ui_009"))
            except Exception:
                logging.exception("Could not save uploaded book page")
                st.error(t("upload_ui_010"))

    st.divider()
    with st.expander(t("upload_ui_005")):
        saved_pages = list_pages()
        if saved_pages:
            choices = {p["page_key"]: p for p in saved_pages}
            selected = st.selectbox(
                t("saved_page_label"), list(choices),
                format_func=lambda k: (
                    t("upload_ui_011").format(v0=choices[k]['book_title'] or t("archive_ui_011"), v1=choices[k]['page_number'], v2=k[:8]))
            )
            if st.button(t("upload_ui_007")):
                try:
                    page, image_bytes = load_page(selected)
                    st.session_state.update({
                        "active_page_key": selected,
                        "uploaded_image": image_bytes,
                        "book_title": page["book_title"],
                        "author": page["author"],
                        "publication_year": page["publication_year"],
                        "page_number": page["page_number"],
                    })
                    st.session_state.pop("preprocess_result", None)
                    st.session_state.pop("preprocess_source_hash", None)
                    st.rerun()
                except Exception:
                    logging.exception("Could not reopen page")
                    st.error(t("upload_ui_012"))
        else:
            st.info(t("upload_ui_008"))


# Page 2: Member 1 real restoration + Member 2 OCR/evidence
elif selected_page == pages[1]:
    from preprocessing import preprocess_image, to_png_bytes, repair_strokes
    import numpy as np
    from suvadi_real_bridge import build_review_tokens
    from suvadi_review_store import get_tokens, add_tokens_if_absent, get_page_info
    import hashlib
    import html
    from tamil_reading import readable_unicode

    st.title(t("restoration_ui_001"))
    st.write(t("restoration_ui_002"))
    image_bytes = st.session_state.get("uploaded_image")
    page_key = st.session_state.get("active_page_key")

    if not image_bytes or not page_key:
        st.warning(t("restoration_ui_003"))
    else:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get("preprocess_source_hash") != image_hash:
            st.session_state["preprocess_source_hash"] = image_hash
            st.session_state.pop("preprocess_result", None)

        if st.button(t("restoration_ui_004"), type="primary"):
            try:
                with st.spinner(t("restoration_ui_021")):
                    processed = preprocess_image(
                        image_bytes, settings=None,
                        page_id=f"page_{st.session_state.get('page_number', 1)}")
                if processed.get("status") != "ok":
                    st.error(t("restoration_ui_022").format(v0=t("error_" + str(processed.get("error_code"))) if "error_" + str(processed.get("error_code")) in UI_TEXT else t("error_unknown")))
                else:
                    st.session_state["preprocess_result"] = processed
                    st.success(t("restoration_ui_023"))
            except Exception:
                logging.exception("Member 1 preprocessing failed")
                st.error(t("restoration_ui_024"))

        result = st.session_state.get("preprocess_result")
        if result is None:
            if get_tokens(page_key):
                st.info(t("restoration_ui_011"))
        else:
            labels = {"gray": t("restoration_ui_005"), "threshold": t("restoration_ui_006"),
            "sharpened": t("restoration_ui_007"),
            "repaired": "எழுத்துக் கோடு சீரமைப்பு"}
            variant_names = [k for k in labels if k in result["variants"]] + ["repaired"]
            selected_variant = st.selectbox(
                t("restoration_ui_008"), variant_names,
                format_func=lambda k: labels[k])

            if selected_variant == "repaired":
                strength = st.select_slider("சீரமைப்பு வலிமை", options=[1, 3, 5], value=3)
                if strength >= 5:
                    st.warning("அதிக வலிமை எழுத்துகளை ஒட்டவைக்கும்; எழுத்தின் வடிவம் மாறலாம்.")
                base = result["variants"].get("gray", result["original_rgb"])
                shown = repair_strokes(np.asarray(base), strength)
            else:
                shown = result["variants"][selected_variant]

            a, b = st.columns(2)
            with a:
                st.subheader(t("original_page_title"))
                st.image(result["original_rgb"], width="stretch")
            with b:
                st.subheader(t("restored_page_title"))
                st.image(shown, width="stretch")

            if st.button(t("restoration_ui_009")):
                st.session_state["enhanced_image"] = to_png_bytes(shown)
                st.success(t("restoration_ui_012"))
            st.download_button(t("restoration_ui_010"),
                to_png_bytes(shown),
                file_name="suvadi_enhanced.png", mime="image/png")

            current = get_tokens(page_key)
            if current:
                st.success(t("restoration_ui_013").format(v0=len(current)))
            else:
                if st.button(t("restoration_ui_014"), type="primary"):
                    try:
                        with st.spinner(t("restoration_ui_025")):
                            records, counts = build_review_tokens(result, page_key)
                            inserted = add_tokens_if_absent(page_key, records)
                        if inserted:
                            st.success(t("restoration_ui_026").format(v0=len(records)))
                            st.caption(t("restoration_ui_027").format(v0=", ".join(f"{t('variant_'+n) if 'variant_'+n in UI_TEXT else n}: {amount}" for n, amount in counts.items())))
                        elif not records:
                            st.warning(t("restoration_ui_029"))
                        else:
                            st.info(t("restoration_ui_030"))
                        st.rerun() if inserted else None
                    except Exception:
                        logging.exception("Member 2 OCR integration failed")
                        st.error(t("restoration_ui_028"))

            current = get_tokens(page_key)
            if current:
                st.divider()
                st.subheader(t("restoration_ui_015"))
                st.info(t("restoration_ui_016"))
                preview = " ".join(t["raw_value"] for t in current)
                st.text_area(t("restoration_ui_017"), value=preview,
                             height=130, disabled=True)
                font_size = st.slider(t("restoration_ui_018"), 18, 38, 25)
                safe = html.escape(readable_unicode(preview))
                st.markdown(
                    "<div style=\"font-family:'Noto Sans Tamil','Nirmala UI',Latha,sans-serif;"
                    f"font-size:{font_size}px;line-height:2;padding:16px;border:1px solid #9996;"
                    "border-radius:10px;overflow-wrap:anywhere;\">"
                    f"{safe}</div>", unsafe_allow_html=True)
                st.caption(t("restoration_ui_019"))
                st.download_button(t("restoration_ui_020"),
                    data=json.dumps({"page_id": page_key,
                                     "source_sha256": (get_page_info(page_key) or {}).get("source_sha256"),
                                     "evidence_results": current},
                                    ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="suvadi_evidence_results.json", mime="application/json")


# Page 3: REAL word evidence review and persistent 5-action decisions
elif selected_page == pages[2]:
    from preprocessing import preprocess_image, crop_word, map_box_to_original
    from suvadi_review_store import get_tokens, save_decision, reset_all, get_audit_events
    from suvadi_real_bridge import review_noise_reasons
    import cv2
    import numpy as np

    st.title(t("review_ui_001"))
    page_key = st.session_state.get("active_page_key")
    tokens = get_tokens(page_key)
    if not page_key:
        st.warning(t("review_ui_002"))
    elif not tokens:
        st.info(t("review_ui_003"))
    else:
        if st.button(t("review_ui_004")):
            reset_all(page_key)
            st.rerun()
        verified = sum(x["status"] == "verified" for x in tokens)
        pending = sum(x["status"] in ("review", "needs_review") for x in tokens)
        counts = st.columns(3)
        counts[0].metric(t("total_words"), len(tokens))
        counts[1].metric(t("verified_count"), verified)
        counts[2].metric(t("pending_count"), pending)
        st.caption(t("review_ui_005"))

        # Lazily reconstruct geometry after restart, if the saved page has been reloaded.
        processed = st.session_state.get("preprocess_result")
        if processed is None and st.session_state.get("uploaded_image"):
            try:
                with st.spinner(t("review_ui_021")):
                    processed = preprocess_image(st.session_state["uploaded_image"])
                if processed.get("status") == "ok":
                    st.session_state["preprocess_result"] = processed
                else:
                    processed = None
            except Exception:
                logging.exception("Could not reconstruct review word crops")
                processed = None

        if processed is not None:
            with st.expander(t("review_ui_016")):
                hide_suspected = st.checkbox(
                    t("review_ui_017"),
                    value=False, key=f"hide_suspicious_boxes_{page_key[:12]}")
                overlay = processed["base_rgb"].copy()
                for token in tokens:
                    if hide_suspected and review_noise_reasons(token):
                        continue
                    box = token.get("bounding_box")
                    if not box:
                        continue
                    x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
                    color = (40, 170, 70) if token["evidence_state"] == "HIGH" else (
                        (255, 160, 0) if token["evidence_state"] == "REVIEW" else (220, 60, 60))
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                st.image(overlay, width="stretch")
                st.caption(t("review_ui_018"))

        st.divider()
        reasons_by_id = {
            token["token_id"]: review_noise_reasons(token) for token in tokens
        }
        suspicious_count = sum(bool(flags) for flags in reasons_by_id.values())
        st.caption(
            t("review_ui_006").format(v0=suspicious_count)
        )
        modes = [t("review_ui_007"), t("review_ui_008"), t("review_ui_009")]
        mode = st.selectbox(
            t("review_ui_010"), modes,
            key=f"ocr_review_filter_{page_key[:12]}"
        )
        if mode == modes[1]:
            view_tokens = [t for t in tokens if reasons_by_id[t["token_id"]]]
        elif mode == modes[2]:
            view_tokens = [t for t in tokens if not reasons_by_id[t["token_id"]]]
        else:
            view_tokens = tokens
        st.caption(t("review_ui_011").format(v0=len(view_tokens), v1=len(tokens)))
        page_size = st.selectbox(t("review_ui_012"), [10, 20, 50])
        num_pages = max(1, (len(view_tokens) + page_size - 1) // page_size)
        page_no = st.number_input(
            t("review_ui_013"), min_value=1, max_value=num_pages,
            value=1,
            key=f"review_page_{page_key[:12]}_{modes.index(mode)}_{page_size}"
        )
        first = (page_no - 1) * page_size
        if not view_tokens:
            st.info(t("review_ui_014"))
        for token in view_tokens[first:first+page_size]:
            tid = token["token_id"]
            unique = f"{page_key[:12]}_{tid}"
            state = token.get("status", "review")
            with st.container(border=True):
                st.subheader(f"📜 {tid}: {token['raw_value']}")
                flags = reasons_by_id.get(tid, [])
                if flags:
                    explanations = {
                        "tiny_box": t("review_ui_022"),
                        "punctuation_only": t("review_ui_023"),
                        "short_low_confidence": t("review_ui_024"),
                    }
                    st.warning(
                        t("review_ui_041")
                        + ", ".join(explanations.get(flag, flag) for flag in flags)
                        + t("review_ui_032")
                    )
                box = token.get("bounding_box")
                if processed is not None and box is not None:
                    left, right = st.columns([1, 2])
                    with left:
                        # Show a *tight* crop: the former 8-pixel margin pulled
                        # neighbouring lines into short, low-resolution word boxes.
                        # Limit on-screen enlargement; never overwrite source pixels.
                        cropped = crop_word(processed["base_rgb"], box, pad=1)
                        if cropped.size:
                            display_width = min(240, max(72, int(cropped.shape[1]) * 2))
                            st.image(
                                cropped,
                                caption=t("review_ui_043"),
                                width=display_width,
                            )
                        # Deskew/crop may change coordinates. Convert the FOUR
                        # bounding-box corners back to the original RGB system.
                        try:
                            import math
                            corners = map_box_to_original(box, processed["metadata"])
                            xs, ys = [p[0] for p in corners], [p[1] for p in corners]
                            h0, w0 = processed["original_rgb"].shape[:2]
                            x0 = max(0, math.floor(min(xs)) - 1)
                            x1 = min(w0, math.ceil(max(xs)) + 1)
                            y0 = max(0, math.floor(min(ys)) - 1)
                            y1 = min(h0, math.ceil(max(ys)) + 1)
                            if x1 > x0 and y1 > y0:
                                original_crop = processed["original_rgb"][y0:y1, x0:x1]
                                display_width = min(240, max(72, int(original_crop.shape[1]) * 2))
                                st.image(
                                    original_crop,
                                    caption=t("review_ui_047"),
                                    width=display_width,
                                )
                        except (ValueError, TypeError, KeyError, OverflowError):
                            logging.exception("Original-page coordinate mapping failed")
                            st.caption(t("review_ui_044"))
                    with right:
                        st.write(t("review_ui_033").format(v0=token['ocr_confidence']))
                        st.write(t("review_ui_034").format(v0=token['evidence_score']))
                        machine_labels = {
                            "HIGH": t("review_ui_025"), "REVIEW": t("review_ui_026"),
                            "UNREADABLE": t("review_ui_027")
                        }
                        st.write(t("review_ui_035").format(v0=machine_labels.get(token['evidence_state'], token['evidence_state'])))
                        st.caption(t("review_ui_036").format(v0=", ".join(t("variant_"+s) if "variant_"+s in UI_TEXT else s for s in token.get("source_variants", []))))
                        st.caption(t("box_position").format(x=box["x"], y=box["y"], w=box["width"], h=box["height"]))
                else:
                    machine_labels = {"HIGH": t("review_ui_025"), "REVIEW": t("review_ui_026"), "UNREADABLE": t("review_ui_027")}
                    st.write(t("review_ui_028").format(v0=token['evidence_score'], v1=machine_labels.get(token['evidence_state'], token['evidence_state'])))
                    st.warning(t("review_ui_029"))
                alternatives = token.get("alternatives", [])
                if alternatives:
                    st.write(t("review_ui_038") + " | ".join(alternatives))

                if state in ("verified", "rejected", "unreadable"):
                    st.info(t("review_ui_039") + {
                        "verified": t("review_ui_045"), "rejected": t("review_ui_046"),
                        "unreadable": t("unreadable_button")}[state])
                    if token.get("final_value"):
                        st.write(t("review_ui_042") + token["final_value"])
                    if st.button(t("review_ui_030"), key=f"redo_{unique}"):
                        save_decision(page_key, tid, "reset")
                        st.rerun()
                    continue

                if state == "needs_review":
                    st.warning(t("review_ui_031"))
                correction = st.text_input(t("review_ui_019"),
                    value=token["raw_value"], key=f"correction_{unique}")
                actions = st.columns(3)
                if actions[0].button(t("accept_button"), key=f"accept_{unique}"):
                    save_decision(page_key, tid, "accept")
                    st.rerun()
                if actions[1].button(t("edit_button"), key=f"edit_{unique}"):
                    if correction.strip():
                        save_decision(page_key, tid, "edit", correction)
                        st.rerun()
                    else:
                        st.error(t("review_ui_040"))
                if actions[2].button(t("reject_button"), key=f"reject_{unique}"):
                    save_decision(page_key, tid, "reject")
                    st.rerun()
                last_actions = st.columns(2)
                if last_actions[0].button(t("unreadable_button"), key=f"unreadable_{unique}"):
                    save_decision(page_key, tid, "unreadable")
                    st.rerun()
                if last_actions[1].button(t("review_later_button"), key=f"flag_{unique}"):
                    save_decision(page_key, tid, "flag")
                    st.rerun()

        with st.expander(t("review_ui_015")):
            events = get_audit_events(page_key)
            if events:
                # Translation is presentation-only; stored field names/actions never change.
                action_labels = {
                    "accept": t("audit_accept"), "edit": t("audit_edit"),
                    "reject": t("audit_reject"), "unreadable": t("audit_unreadable"),
                    "flag": t("audit_flag"), "reset": t("audit_reset"),
                }
                column_labels = {
                    "event_id": t("audit_event_id"),
                    "page_key": t("audit_page_key"),
                    "token_id": t("audit_token_id"),
                    "reviewer_action": t("audit_action"),
                    "original_value": t("audit_original"),
                    "corrected_value": t("audit_corrected"),
                    "timestamp": t("audit_time"),
                }
                display_events = []
                for event in events:
                    item = dict(event)
                    if item.get("reviewer_action") in action_labels:
                        item["reviewer_action"] = action_labels[item["reviewer_action"]]
                    if item.get("corrected_value") is None:
                        item["corrected_value"] = "—"
                    display_events.append({column_labels.get(k, k): v for k, v in item.items()})
                st.dataframe(display_events, width="stretch", hide_index=True)
            else:
                st.info(t("review_ui_020"))


# Page 4: verified text is ONLY human-approved words
elif selected_page == pages[3]:
    from suvadi_review_store import get_tokens
    st.title(t("verified_ui_001"))
    tokens = get_tokens(st.session_state.get("active_page_key"))
    if not tokens:
        st.info(t("verified_ui_002"))
    else:
        approved = [t for t in tokens if t.get("status") == "verified"
                    and t.get("review_action") in ("accept", "edit")
                    and t.get("final_value")]
        rejected = sum(t["status"] == "rejected" for t in tokens)
        unreadable = sum(t["status"] == "unreadable" for t in tokens)
        pending = len(tokens) - len(approved) - rejected - unreadable
        a,b,c = st.columns(3)
        a.metric(t("verified_ui_003"), len(approved))
        b.metric(t("verified_ui_004"), rejected)
        c.metric(t("verified_ui_005"), unreadable)
        if pending:
            st.warning(t("verified_ui_006").format(v0=pending))
        if rejected or unreadable or pending:
            st.info(t("verified_ui_007"))
        if approved:
            verified_text = " ".join(t["final_value"] for t in approved)
            st.text_area(t("verified_ui_008"), verified_text,
                         height=200, disabled=True)
            st.download_button(t("verified_ui_009"),
                verified_text.encode("utf-8"), file_name="suvadi_verified_text.txt",
                mime="text/plain")
        else:
            st.info(t("verified_ui_010"))


# Page 5: Existing archive remains unchanged; live reviewed tokens are used
elif selected_page == pages[4]:
    from archive import load_archive, save_archive_record, search_archive
    from suvadi_review_store import get_tokens, get_page_info
    st.title(t("archive_ui_001"))
    st.caption(t("archive_ui_002"))
    tokens = get_tokens(st.session_state.get("active_page_key"))
    approved = [t for t in tokens if t["status"] == "verified" and t.get("final_value")]
    pending = sum(t["status"] in ("review", "needs_review") for t in tokens)
    if approved:
        if pending:
            st.warning(t("archive_ui_008"))
        st.caption(t("archive_ui_005"))
        if st.button(t("archive_ui_006"), disabled=bool(pending)):
            try:
                page_key = st.session_state.get("active_page_key")
                page_info = get_page_info(page_key)
                if page_info is None:
                    st.error(t("archive_ui_missing_page"))
                else:
                    source_sha256 = page_info.get("source_sha256")
                    save_archive_record(
                        page_info["book_title"],
                        page_info["author"],
                        page_info["page_number"],
                        tokens,
                        page_key=page_info["page_key"],
                        source_sha256=source_sha256,
                    )
                    st.success(t("archive_ui_009"))
                    if source_sha256 is None:
                        st.warning(t("archive_ui_legacy_hash"))
            except Exception:
                logging.exception("Could not save verified page to archive")
                st.error(t("archive_ui_save_failed"))
    else:
        st.info(t("archive_ui_007"))
    st.divider()
    query = st.text_input(t("archive_ui_003"))
    records = search_archive(query)
    st.write(t("archive_ui_004").format(v0=len(records)))
    for record in reversed(records):
        with st.expander(t("archive_ui_010").format(v0=record['book_title'], v1=record['page_number'])):
            st.write(t("archive_ui_013") + record.get("author", ""))
            st.write(t("archive_ui_014") + record.get("verified_text", ""))
            st.caption(t("archive_ui_015") + str(record.get("record_id", "")))


# Page 6: actual current-page review stats, NOT OCR accuracy claims
elif selected_page == pages[5]:
    from archive import load_archive
    from suvadi_review_store import get_tokens
    st.title(t("dashboard_ui_001"))
    st.info(t("dashboard_ui_002"))
    tokens = get_tokens(st.session_state.get("active_page_key"))
    archived = load_archive()
    verified = sum(t["status"] == "verified" for t in tokens)
    rejected = sum(t["status"] == "rejected" for t in tokens)
    unreadable = sum(t["status"] == "unreadable" for t in tokens)
    pending = sum(t["status"] in ("review", "needs_review") for t in tokens)
    a,b,c = st.columns(3)
    a.metric(t("dashboard_ui_003"), len(archived))
    b.metric(t("dashboard_ui_004"), len(tokens))
    c.metric(t("verified_count"), verified)
    d,e,f = st.columns(3)
    d.metric(t("verified_ui_004"), rejected)
    e.metric(t("verified_ui_005"), unreadable)
    f.metric(t("dashboard_ui_005"), pending)
    if tokens:
        completed = verified + rejected + unreadable
        st.progress(completed / len(tokens),
                    text=t("dashboard_ui_008").format(v0=100 * completed / len(tokens)))
        st.bar_chart({
            t("dashboard_ui_009"): [t("dashboard_ui_011"), t("dashboard_ui_012"), t("dashboard_ui_013"), t("dashboard_ui_014")],
            t("dashboard_ui_010"): [verified, rejected, unreadable, pending],
        }, x=t("dashboard_ui_009"), y=t("dashboard_ui_010"), horizontal=True)
    else:
        st.info(t("dashboard_ui_007"))
    st.caption(t("dashboard_ui_006"))
