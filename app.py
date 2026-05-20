import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types as genai_types
import fitz  # PyMuPDF — replaces pdf2image + Poppler
import json
import os
import io
import base64
import hashlib
import tempfile
import time
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# ---------------- CONFIGURATION ----------------
CHECKPOINT_DIR = "checkpoints"


def _safe_response_text(response) -> str | None:
    try:
        txt = response.text
        if txt is not None and txt.strip():
            return txt
    except Exception:
        pass

    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "thought", False):
                    continue
                t = getattr(part, "text", None)
                if t and t.strip():
                    return t
    except Exception:
        pass

    try:
        reason = response.candidates[0].finish_reason
        print(f"  [empty response] finish_reason={reason}")
    except Exception:
        pass
    return None


def _call_gemini(client, model_name: str, contents: list, cfg) -> str | None:
    response = client.models.generate_content(
        model=model_name, contents=contents, config=cfg,
    )
    return _safe_response_text(response)


def _make_page_configs():
    cfgs = []

    try:
        cfgs.append(genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=65536))
    except Exception as _e:
        print(f"[cfg] plain failed: {_e}")

    try:
        cfgs.append(genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=65536,
            response_mime_type="application/json"))
    except Exception as _e:
        print(f"[cfg] json-mime failed: {_e}")

    for budget in [1024, 8192, 24576]:
        try:
            cfgs.append(genai_types.GenerateContentConfig(
                temperature=0, max_output_tokens=65536,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=budget)))
        except Exception as _e:
            print(f"[cfg] thinking budget={budget} failed: {_e}")

    if not cfgs:
        cfgs = [genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=65536)]

    return cfgs

st.set_page_config(
    page_title="Ultra-Fast Extraction: PDF to JSON",
    page_icon="⚡",
    layout="wide"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

FIELD_ORDER = [
    "questionid", "question", "option1", "option2", "option3", "option4",
    "Answer", "Explanation", "course", "subjectname", "chapter", "practice",
    "subtopic", "medium", "difficulty", "question_type", "previous_year",
    "marks", "class", "book", "question_bucket",
]

# ================================================================
# PyMuPDF PDF HELPERS
# ================================================================

EXTRACTION_DPI = 350

def pdf_page_to_png_bytes(pdf_path: str, page_num: int, dpi: int = EXTRACTION_DPI) -> bytes:
    """Render PDF page to PNG. FIX: guard against out-of-range page_num."""
    doc = fitz.open(pdf_path)
    # ── FIX 1: guard page_num out of range ──
    if page_num >= len(doc):
        doc.close()
        raise IndexError(f"Page index {page_num} out of range — document has {len(doc)} page(s)")
    page = doc[page_num]
    zoom = dpi / 72
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    try:
        import cv2, numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        img = cv2.filter2D(img, -1, kernel)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        _, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return buf.tobytes()
    except Exception:
        return pix.tobytes("png")


def compute_template_image_hashes(pdf_path: str) -> set:
    try:
        doc = fitz.open(pdf_path)
        hash_page_count: dict[str, int] = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            seen_this_page: set[str] = set()
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    raw = doc.extract_image(xref)["image"]
                    h = hashlib.md5(raw).hexdigest()
                    if h not in seen_this_page:
                        seen_this_page.add(h)
                        hash_page_count[h] = hash_page_count.get(h, 0) + 1
                except Exception:
                    pass
        doc.close()
        return {h for h, cnt in hash_page_count.items() if cnt >= 3}
    except Exception:
        return set()


# ================================================================
# IMAGE EXTRACTION
# ================================================================

_IMG_FIELDS = ["question", "Explanation", "option1", "option2", "option3", "option4"]

_DIAGRAM_KWS = [
    'figure', 'fig.', 'fig ', 'diagram', 'image', 'shown', 'given', 'above', 'below',
    'graph', 'circuit', 'spinner', 'marble', 'bag', 'chart', 'map', 'plot',
    'shape', 'triangle', 'rectangle', 'circle', 'polygon', 'structure', 'illustration',
    'चित्र', 'आकृति', 'दिया गया', 'नीचे', 'ऊपर', 'आरेख', 'ग्राफ', 'परिपथ',
]


def extract_page_embedded_images(pdf_path: str, page_num: int,
                                  skip_hashes: set | None = None) -> list[dict]:
    """FIX: guard page_num out of range."""
    doc = fitz.open(pdf_path)
    # ── FIX 1 (also here): guard page_num ──
    if page_num >= len(doc):
        doc.close()
        return []
    page = doc[page_num]
    seen_xrefs: set[int] = set()

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref > 0:
            seen_xrefs.add(xref)

    try:
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref", 0)
            if xref and xref > 0:
                seen_xrefs.add(xref)
    except Exception:
        pass

    xref_ypos: dict[int, float] = {}
    try:
        for info in page.get_image_info(hashes=False):
            xref = info.get("xref", 0)
            if xref and xref > 0:
                bbox = info.get("bbox", [0, 0, 0, 0])
                xref_ypos[xref] = float(bbox[1]) if len(bbox) >= 4 else 0.0
    except Exception:
        pass

    images = []
    for xref in seen_xrefs:
        try:
            img_data = doc.extract_image(xref)
            raw      = img_data["image"]
            ext      = img_data.get("ext", "png").lower()
            w        = img_data.get("width",  0)
            h        = img_data.get("height", 0)
            cs_n     = img_data.get("colorspace", 3)

            if skip_hashes and hashlib.md5(raw).hexdigest() in skip_hashes:
                continue
            if w < 40 or h < 30:
                continue
            if _is_useless_image(raw):
                continue

            if ext in ("jpg", "jpeg") and cs_n == 3:
                mime = "image/jpeg"
                b64  = base64.b64encode(raw).decode()
            elif ext == "png" and cs_n in (1, 3):
                mime = "image/png"
                b64  = base64.b64encode(raw).decode()
            else:
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n != 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                raw  = pix.tobytes("png")
                mime = "image/png"
                b64  = base64.b64encode(raw).decode()

            images.append({
                "mime_type": mime,
                "image_base64": b64,
                "data_uri": f"data:{mime};base64,{b64}",
                "y_pos": xref_ypos.get(xref, 9999.0),
            })
        except Exception:
            pass

    doc.close()
    return images


def _is_useless_image(raw_bytes: bytes) -> bool:
    try:
        pix = fitz.Pixmap(raw_bytes)
        if pix.width < 10 or pix.height < 10:
            return True
        samples = pix.samples
        n = pix.n
        total = len(samples) // n
        step = max(1, total // 400)
        dark_count = 0
        sampled = 0
        for i in range(0, total, step):
            px = samples[i*n : i*n + min(3, n)]
            if len(px) >= 3:
                brightness = (px[0] + px[1] + px[2]) / 3
                if brightness < 180:
                    dark_count += 1
                sampled += 1
        if sampled == 0:
            return False
        dark_ratio = dark_count / sampled
        if dark_ratio < 0.005:
            return True
        if dark_ratio > 0.98:
            return True
        return False
    except Exception:
        return False


def _inflate_rect(r: fitz.Rect, d: float) -> fitz.Rect:
    return fitz.Rect(r.x0 - d, r.y0 - d, r.x1 + d, r.y1 + d)


def _cluster_rects(rects: list, gap: int = 25) -> list:
    clusters: list[fitz.Rect] = []
    for rect in rects:
        expanded = _inflate_rect(rect, gap)
        merged = False
        for i, cluster in enumerate(clusters):
            if expanded.intersects(cluster):
                clusters[i] = clusters[i] | rect
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(rect))
    return clusters


def extract_vector_diagram_regions(pdf_path: str, page_num: int) -> list[dict]:
    try:
        doc  = fitz.open(pdf_path)
        # ── FIX 1 (also here): guard page_num ──
        if page_num >= len(doc):
            doc.close()
            return []
        page = doc[page_num]
        crop_mat = fitz.Matrix(3.0, 3.0)
        images: list[dict] = []
        captured: list[fitz.Rect] = []

        def _already_covered(r: fitz.Rect) -> bool:
            return any(
                cap.intersect(r).get_area() / max(r.get_area(), 1) > 0.5
                for cap in captured
            )

        def _render(clip: fitz.Rect, y_pos: float):
            clip = clip & page.rect
            if clip.width < 10 or clip.height < 10:
                return
            pix = page.get_pixmap(matrix=crop_mat, clip=clip, colorspace=fitz.csRGB)
            raw = pix.tobytes("png")
            if _is_useless_image(raw):
                return
            b64 = base64.b64encode(raw).decode()
            images.append({
                "mime_type": "image/png",
                "image_base64": b64,
                "data_uri": f"data:image/png;base64,{b64}",
                "y_pos": y_pos,
            })
            captured.append(clip)

        drawings = page.get_drawings()
        if drawings:
            raw_rects = [fitz.Rect(d["rect"]) for d in drawings
                         if d.get("rect") and max(fitz.Rect(d["rect"]).width,
                                                   fitz.Rect(d["rect"]).height) >= 5]

            clusters: list[fitz.Rect] = []
            cluster_counts: list[int] = []
            for rect in raw_rects:
                exp = _inflate_rect(rect, 20)
                merged = False
                for i, c in enumerate(clusters):
                    if exp.intersects(c):
                        clusters[i] = clusters[i] | rect
                        cluster_counts[i] += 1
                        merged = True
                        break
                if not merged:
                    clusters.append(fitz.Rect(rect))
                    cluster_counts.append(1)

            for cr, cnt in zip(clusters, cluster_counts):
                if cr.width < 15 or cr.height < 15:
                    continue
                if cr.get_area() < 300:
                    continue
                if cnt < 2:
                    continue
                aspect = cr.width / max(cr.height, 1)
                if aspect > 3.0 and cr.height < 60:
                    continue
                if _already_covered(cr):
                    continue
                clip = fitz.Rect(
                    max(0, cr.x0 - 30), max(0, cr.y0 - 8),
                    min(page.rect.width,  cr.x1 + 30),
                    min(page.rect.height, cr.y1 + 8)
                )
                _render(clip, cr.y0)

        try:
            for annot in page.annots():
                ar = fitz.Rect(annot.rect)
                if ar.width >= 15 and ar.height >= 15 and not _already_covered(ar):
                    _render(_inflate_rect(ar, 8), ar.y0)
        except Exception:
            pass

        doc.close()
        images.sort(key=lambda x: x.get("y_pos", 0))
        return images
    except Exception:
        return []


def extract_all_page_images(pdf_path: str, page_num: int,
                             skip_hashes: set | None = None) -> list[dict]:
    raster = extract_page_embedded_images(pdf_path, page_num, skip_hashes)
    vector = extract_vector_diagram_regions(pdf_path, page_num)
    deduped_vector = [v for v in vector
                      if not any(abs(v.get("y_pos", 0) - r.get("y_pos", 9999)) < 30
                                 for r in raster)]
    combined = raster + deduped_vector
    combined.sort(key=lambda x: x.get("y_pos", 9999))
    return combined


# ================================================================
# DIAGRAM INJECTION HELPERS
# ================================================================

def _normalize_diagram_refs(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\bdiagram\[(\d+)\]',               r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[diagram_(\d+)\]',                 r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[DIAGRAM(\d+)\]',                  r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[image[_ ]?(\d+)\]',               r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\[)\bdiagram\s+(\d+)\b(?!\])', r'[DIAGRAM_\1]', text, flags=re.IGNORECASE)
    return text


def _img_tag(data_uri: str, width: int = 340) -> str:
    return (f'<img src="{data_uri}" '
            f'style="max-width:{width}px;width:100%;height:auto;'
            f'display:block;margin:8px auto;border-radius:2px"/>')


def _cleanup_placeholders(q: dict) -> dict:
    for field in _IMG_FIELDS:
        if field in q and q[field]:
            q[field] = re.sub(r'\[DIAGRAM_?\d*\]', '', q[field])
            q[field] = re.sub(r'\[DIAGRAM\]',       '', q[field])
    return q


def inject_diagrams_into_question(q: dict, page_images: list[dict]) -> dict:
    for field in _IMG_FIELDS:
        if field in q and q[field]:
            q[field] = _normalize_diagram_refs(q[field])

    if not page_images:
        q.pop("diagram_placements", None)
        q.pop("diagram_image_indices", None)
        return _cleanup_placeholders(q)

    injected_indices: set[int] = set()
    for field in _IMG_FIELDS:
        if field not in q or not q[field]:
            continue
        for match in re.findall(r'\[DIAGRAM_(\d+)\]', q[field]):
            idx = int(match)
            if 0 <= idx < len(page_images):
                tag = _img_tag(page_images[idx]["data_uri"])
                q[field] = q[field].replace(f'[DIAGRAM_{idx}]', tag)
                injected_indices.add(idx)
            else:
                q[field] = q[field].replace(f'[DIAGRAM_{idx}]', '')

    placements       = q.pop("diagram_placements", []) or []
    indices_fallback = q.pop("diagram_image_indices", []) or []

    for p in placements:
        idx      = p.get("image_index", 0)
        field    = p.get("field", "Explanation")
        position = p.get("position", "end")
        if 0 <= idx < len(page_images) and field in q and idx not in injected_indices:
            tag = _img_tag(page_images[idx]["data_uri"])
            if position == "start":
                q[field] = tag + "\n" + (q.get(field) or "")
            elif position == "replace" and "[DIAGRAM]" in q.get(field, ""):
                q[field] = q[field].replace("[DIAGRAM]", tag)
            else:
                q[field] = (q.get(field) or "") + "\n" + tag
            injected_indices.add(idx)

    q_text   = (q.get("question", "")    or "").lower()
    exp_text = (q.get("Explanation", "") or "").lower()
    for idx in indices_fallback:
        if 0 <= idx < len(page_images) and idx not in injected_indices:
            tag = _img_tag(page_images[idx]["data_uri"], width=300)
            if any(kw in q_text for kw in _DIAGRAM_KWS):
                q["question"]    = (q.get("question")    or "") + f"\n{tag}"
                injected_indices.add(idx)
            elif any(kw in exp_text for kw in _DIAGRAM_KWS):
                q["Explanation"] = (q.get("Explanation") or "") + f"\n{tag}"
                injected_indices.add(idx)

    return _cleanup_placeholders(q)

# ================================================================
# LIVE PREVIEW HELPERS
# ================================================================

def _fmt_latex(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
    text = re.sub(r'<img[^>]*>', '[📷]', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\\\\\((.+?)\\\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\\\[(.+?)\\\\\]', r'$$\1$$', text, flags=re.DOTALL)
    return text


_LADDER_DIVISOR_RE = re.compile(r'^\s*\d+\s*\|')
_LADDER_SEP_RE     = re.compile(r'^\s*\|[_\s]+$')
_LADDER_EMPTY_DIV  = re.compile(r'^\s*\|')
_LADDER_FINAL_RE   = re.compile(r'^\s+\d[\d\s]+$')
_LADDER_ROW_PARSE  = re.compile(r'^(\s*\d*)\s*\|\s*(.+)$')


def _lcm_ladder_to_html(ladder_lines: list) -> str:
    rows       = []
    final_nums = None

    for line in ladder_lines:
        stripped = line.strip()
        if _LADDER_SEP_RE.match(stripped):
            continue
        m = _LADDER_ROW_PARSE.match(stripped)
        if m:
            rows.append((m.group(1).strip(), m.group(2)))
        elif _LADDER_FINAL_RE.match(line):
            final_nums = line.strip()

    if not rows:
        return ''

    VL = '2px solid #111'
    HL = '1.5px solid #111'

    S_D = (f'border:none;border-right:{VL};border-bottom:{HL};'
           f'text-align:right;padding:3px 6px 3px 4px;vertical-align:middle;'
           f'font-weight:bold;min-width:20px')
    S_N = (f'border:none;border-bottom:{HL};'
           f'padding:3px 10px 3px 8px;white-space:pre;vertical-align:middle')

    tbl = ('<table class="lcm-ladder-t" style="border-collapse:collapse;'
           'font-family:\'Courier New\',Courier,monospace;font-size:14px;'
           'line-height:1.9;margin:8px 0">')

    for div, nums in rows:
        de = div.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        ne = nums.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        tbl += f'<tr><td style="{S_D}">{de}</td><td style="{S_N}">{ne}</td></tr>'

    if final_nums is not None:
        fe = final_nums.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        SF_D = 'border:none;padding:3px 6px 3px 4px;min-width:20px'
        SF_N = 'border:none;padding:3px 10px 3px 8px;white-space:pre;vertical-align:middle'
        tbl += f'<tr><td style="{SF_D}"></td><td style="{SF_N}">{fe}</td></tr>'

    tbl += '</table>'
    return tbl


def _md_table_to_html(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if '|' in line and stripped.startswith('|') and not _LADDER_SEP_RE.match(stripped):
            table_lines = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            rows = [r for r in table_lines if not re.match(r'^\s*\|[\s\-|:]+\|\s*$', r)]
            if rows:
                html = '<table class="qtable" style="border-collapse:collapse;margin:8px 0">'
                for ri, row in enumerate(rows):
                    cells = [c.strip() for c in row.strip().strip('|').split('|')]
                    if ri == 0:
                        html += ('<tr>' + ''.join(
                            f'<th style="border:1px solid #333;padding:5px 10px;'
                            f'background:#dbeafe;font-weight:bold;text-align:center">{c}</th>'
                            for c in cells) + '</tr>')
                    else:
                        html += ('<tr>' + ''.join(
                            f'<td style="border:1px solid #333;padding:5px 10px;'
                            f'text-align:left;vertical-align:middle">{c}</td>'
                            for c in cells) + '</tr>')
                html += '</table>'
                out.append(html)
            continue

        if _LADDER_DIVISOR_RE.match(stripped):
            ladder_lines = []
            while i < len(lines):
                s = lines[i]
                ss = s.strip()
                if (_LADDER_DIVISOR_RE.match(ss) or
                        _LADDER_SEP_RE.match(ss) or
                        _LADDER_EMPTY_DIV.match(ss) or
                        (ladder_lines and _LADDER_FINAL_RE.match(s))):
                    ladder_lines.append(s)
                    i += 1
                elif ss == '' and ladder_lines:
                    break
                else:
                    break
            if len(ladder_lines) >= 2:
                out.append(_lcm_ladder_to_html(ladder_lines))
            else:
                out.extend(ladder_lines)
            continue

        out.append(line)
        i += 1
    return '\n'.join(out)


def _normalize_html_table(table_html: str) -> str:
    head = table_html[:120].lower()
    if 'lcm-ladder-t' in head:
        return table_html
    if 'class=' not in head:
        return table_html.replace('<table', '<table class="ltable"', 1)
    if 'qtable' not in head and 'ltable' not in head and 'lcm-table' not in head:
        return re.sub(r'class="', 'class="ltable ', table_html, count=1, flags=re.IGNORECASE)
    return table_html


_KEEP_RE = re.compile(
    r'(<img\b[^>]*?>'
    r'|<table\b[^>]*?>.*?</table\s*>'
    r'|<pre\b[^>]*?>.*?</pre\s*>)',
    re.IGNORECASE | re.DOTALL
)


def _clean_for_html(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
    text = _md_table_to_html(text)
    segments = _KEEP_RE.split(text)
    result = []
    for seg in segments:
        if _KEEP_RE.match(seg):
            if seg.lower().startswith('<table'):
                seg = _normalize_html_table(seg)
            result.append(seg)
        else:
            s = seg
            s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
            s = re.sub(r'<[^>]+>', '', s)
            s = re.sub(r'\\\\\((.+?)\\\\\)', r'\\(\1\\)', s, flags=re.DOTALL)
            s = re.sub(r'\\\\\[(.+?)\\\\\]', r'\\[\1\\]', s, flags=re.DOTALL)
            s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            s = s.replace('\n', '<br>')
            result.append(s)
    return ''.join(result)


def _build_preview_html(questions: list) -> str:
    if not questions:
        return '<p style="color:#888;padding:16px;font-family:sans-serif">Waiting for questions…</p>'
    body_parts = []
    for q in questions:
        qid  = q.get('questionid', '?')
        qtxt = _clean_for_html(str(q.get('question', '')))
        opts_html = ''
        for i in range(1, 5):
            opt = q.get(f'option{i}', '')
            if opt:
                lbl = chr(64 + i)
                opts_html += (
                    f'<div class="opt">'
                    f'<span class="lbl">({lbl})</span>&nbsp;{_clean_for_html(str(opt))}'
                    f'</div>'
                )
        ans = q.get('Answer', '')
        exp = q.get('Explanation', '')
        ans_html = f'<div class="ans">&#x2705; <b>Answer:</b> {ans}</div>' if ans else ''
        exp_html = (
            f'<div class="exp">&#x1F4A1; <b>Explanation:</b> {_clean_for_html(str(exp))}</div>'
            if exp else ''
        )
        body_parts.append(
            f'<div class="qblock">'
            f'<div class="qtext"><b>Q{qid}.</b>&nbsp;{qtxt}</div>'
            f'<div class="opts">{opts_html}</div>'
            f'{ans_html}{exp_html}'
            f'</div>'
        )
    body = '\n'.join(body_parts)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">\n'
        '<script>\n'
        'window.MathJax = {\n'
        '  tex: {\n'
        '    inlineMath: [["\\\\(","\\\\)"]],\n'
        '    displayMath: [["\\\\[","\\\\]"]],\n'
        '    processEscapes: true\n'
        '  },\n'
        '  options: { skipHtmlTags: ["script","noscript","style","textarea"] }\n'
        '};\n'
        '</script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>\n'
        '<style>\n'
        'body{font-family:"Times New Roman",serif;font-size:14px;line-height:1.8;color:#111;background:#fff;padding:12px;margin:0}\n'
        '.qblock{margin-bottom:20px;padding:10px 14px;border-left:3px solid #2563eb;background:#f8fafc;border-radius:3px}\n'
        '.qtext{margin-bottom:8px}.opts{margin-left:18px}.opt{margin:4px 0}\n'
        '.lbl{font-weight:bold;color:#374151}\n'
        '.ans{margin-top:8px;color:#15803d;font-size:13px}\n'
        '.exp{margin-top:6px;color:#374151;font-size:13px}\n'
        'table{border-collapse:collapse;margin:8px 0;max-width:100%;font-size:13px;font-family:"Times New Roman",serif}\n'
        'table td,table th{border:1px solid #444;padding:5px 10px;vertical-align:middle;text-align:center}\n'
        'table th{background:#dbeafe;font-weight:bold;color:#1e3a8a}\n'
        'table tr:nth-child(even) td{background:#f8fafc}\n'
        'table.lcm-table td:first-child,.lcm-div{border-right:2.5px solid #000!important;background:#eef2ff;font-weight:bold;min-width:30px;text-align:center}\n'
        'table.lcm-table td{min-width:38px;text-align:center;padding:4px 8px}\n'
        'table.lcm-table tr:last-child td{border-top:2px solid #111}\n'
        'table.qtable td,table.qtable th,table.ltable td,table.ltable th{text-align:left}\n'
        'table.match-col td:first-child,table.match-col th:first-child{border-right:2px solid #555}\n'
        'img{max-width:100%;height:auto;display:block;margin:6px auto;border-radius:2px}\n'
        '.img-row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;margin:6px 0}\n'
        '.img-row img{max-width:calc(50% - 4px);flex:0 1 auto;margin:0}\n'
        '.compare-wrap{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;margin:8px 0}\n'
        '.compare-wrap>*{flex:1 1 auto;min-width:180px}\n'
        'mjx-container[display="true"]{display:block;margin:6px 0;overflow-x:auto}\n'
        '.MathJax{font-size:1em!important}\n'
        'table.lcm-ladder-t{border-collapse:collapse}'
        '\n'
        '</style></head><body>\n'
        + body +
        '\n</body></html>'
    )


# ================================================================
# SECTION CONFIG HELPERS
# ================================================================
def build_section_page_map(section_configs, total_pages):
    page_marks = [""] * total_pages
    for sec in section_configs:
        start = sec.get("start_page", 1)
        end   = sec.get("end_page", total_pages)
        marks = str(sec.get("marks", "")).strip()
        for p in range(start - 1, min(end, total_pages)):
            page_marks[p] = marks
    return page_marks

# ================================================================
# CHECKPOINT HELPERS
# ================================================================
def get_checkpoint_path(filename):
    base = os.path.splitext(filename)[0]
    return os.path.join(CHECKPOINT_DIR, f"{base}_checkpoint.json")


def save_checkpoint(filename, page_results, total_pages):
    checkpoint = {
        "total_pages": total_pages,
        "completed_pages": {
            str(i): data for i, data in enumerate(page_results)
            if data is not None and len(data) > 0
        }
    }
    with open(get_checkpoint_path(filename), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)


def load_checkpoint(filename, total_pages):
    path = get_checkpoint_path(filename)
    if not os.path.exists(path):
        return [None] * total_pages, 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        if checkpoint.get("total_pages", 0) != total_pages:
            return [None] * total_pages, 0
        page_results = [None] * total_pages
        for idx_str, data in checkpoint.get("completed_pages", {}).items():
            page_results[int(idx_str)] = data
        done = sum(1 for r in page_results if r is not None)
        return page_results, done
    except Exception:
        return [None] * total_pages, 0


def delete_checkpoint(filename):
    path = get_checkpoint_path(filename)
    if os.path.exists(path):
        os.remove(path)

# ================================================================
# ENHANCED POST-PROCESSING HELPERS
# ================================================================
def remove_cite_tags(text):
    if not text or not isinstance(text, str):
        return text
    return re.sub(r'\s*\[cite\s*:\s*\d+\]', '', text).strip()


def convert_dollar_to_latex(text):
    if not text or not isinstance(text, str):
        return text
    text = re.sub(r'\$\$(.+?)\$\$',
                  lambda m: '\\\\[' + m.group(1) + '\\\\]',
                  text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
                  lambda m: '\\\\(' + m.group(1) + '\\\\)',
                  text, flags=re.DOTALL)
    return text


def remove_base64_images(text):
    if not text or not isinstance(text, str):
        return text
    text = re.sub(
        r'(?<!src=["\'])(?<!src=)data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}',
        '', text
    )
    return text


def newline_to_br(text):
    if not text or not isinstance(text, str):
        return text
    if '<table' in text.lower():
        return text
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = text.replace('\n', '<br>')
    return text


def clean_explanation_prefix(text):
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    prefix_pattern = re.compile(
        r'^(?:Ans(?:wer)?\.?\s*:?\s*(?:\([A-Da-d]\))?\s*|Sol(?:ution)?\.?\s*:?\s*)',
        re.IGNORECASE
    )
    hindi_prefix_pattern = re.compile(
        r'^(?:उत्तर\s*:?\s*|हल\s*:?\s*|समाधान\s*:?\s*|व्याख्या\s*:?\s*)',
        re.UNICODE
    )
    for _ in range(3):
        cleaned = prefix_pattern.sub('', text).strip()
        cleaned = hindi_prefix_pattern.sub('', cleaned).strip()
        if cleaned == text:
            break
        if not cleaned:
            break
        text = cleaned
    return text


# ================================================================
# FIX: ENHANCED previous_year EXTRACTION FROM QUESTION TEXT
# ================================================================

_PY_INLINE_PATTERNS = [
    re.compile(
        r'[-–—]?\s*[\(\[]\s*'
        r'((?:Exercise|Ex|Miscellaneous\s+Exercise|Misc\.?\s*Exercise|Miscellaneous|'
        r'Example|Ex\.?|NCERT|PYQ|Previous\s+Year|Exemplar)\s*[-–]?\s*[\d\.]+(?:[-–]\d+)?'
        r'(?:\s*[-–]\s*\d+)*)\s*[\)\]]',
        re.IGNORECASE
    ),
    re.compile(
        r'[-–—]?\s*[\(\[]?\s*\b((19|20)\d{2})\b\s*[\)\]]?',
        re.IGNORECASE
    ),
    re.compile(
        r'[-–—]?\s*[\(\[]?\s*'
        r'((?:JEE\s*(?:Main|Advanced|Mains)?|NEET|AIIMS|CBSE|ICSE|BITSAT|MHT[- ]?CET|'
        r'WBJEE|KCET|UPSEE|VITEEE|COMEDK|NDA|CDS|UPSC|Board)\s*[-–]?\s*(?:(19|20)\d{2})?)\s*'
        r'[\)\]]?',
        re.IGNORECASE
    ),
    re.compile(
        r'[-–—]\s*'
        r'((?:Miscellaneous\s+Exercise|Misc\.?\s*Exercise|NCERT\s+Exemplar|'
        r'Exercise|Example|Ex\.?)\s*[-–]?\s*[\d\.]+(?:[-–]\d+)*)',
        re.IGNORECASE
    ),
]

_PY_END_TAG_RE = re.compile(
    r'\s*[-–—]?\s*[\(\[]\s*'
    r'((?:Exercise|Ex\.?|Misc(?:ellaneous)?\s*Exercise|Example|NCERT|PYQ|'
    r'JEE\s*(?:Main|Advanced|Mains)?|NEET|AIIMS|CBSE|ICSE|Board|'
    r'(?:19|20)\d{2})'
    r'[\s\d\.,-]*)\s*[\)\]]\s*$',
    re.IGNORECASE
)


def extract_previous_year_from_text(text: str) -> tuple[str, str]:
    if not text or not isinstance(text, str):
        return "", text

    m = _PY_END_TAG_RE.search(text)
    if m:
        ref = m.group(1).strip()
        cleaned = text[:m.start()].strip()
        cleaned = re.sub(r'[\s\-–—]+$', '', cleaned)
        return ref, cleaned

    for pattern in _PY_INLINE_PATTERNS:
        m = pattern.search(text)
        if m:
            ref = m.group(1).strip() if m.lastindex and m.group(1) else m.group(0).strip()
            if m.start() >= max(0, len(text) - 120):
                cleaned = text[:m.start()].strip()
                cleaned = re.sub(r'[\s\-–—]+$', '', cleaned)
                if ref:
                    return ref, cleaned

    return "", text


def fix_previous_year_field(py_val: str) -> str:
    if not py_val or not isinstance(py_val, str):
        return ""
    val = py_val.strip()
    garbage = {
        'null', 'none', 'n/a', 'not applicable', 'not found',
        'not available', 'unknown', 'na', '-', '', 'no'
    }
    if val.lower() in garbage:
        return ""

    if re.search(r'\b(19|20)\d{2}\b', val):
        return val
    if re.search(
        r'(ncert|exercise|example|ex\.?\s*\d|miscellaneous|misc|exemplar|'
        r'jee|neet|aiims|cbse|icse|pyq|previous\s*year)',
        val, re.IGNORECASE
    ):
        return val
    return ""


# ================================================================
# SUBTOPIC AUTO-FILL
# ================================================================

_SUBTOPIC_HINDI_MAP = {
    "Laws of Motion": "गति के नियम",
    "Work Energy Power": "कार्य ऊर्जा और शक्ति",
    "Gravitation": "गुरुत्वाकर्षण",
    "Current Electricity": "विद्युत धारा",
    "Electrostatics": "स्थिरवैद्युतिकी",
    "Magnetism": "चुंबकत्व",
    "Waves": "तरंगें",
    "Optics": "प्रकाशिकी",
    "Thermodynamics": "ऊष्मागतिकी",
    "Semiconductors": "अर्धचालक",
    "Atoms and Nuclei": "परमाणु और नाभिक",
    "Friction": "घर्षण",
    "Rotational Motion": "घूर्णन गति",
    "Oscillations": "दोलन",
    "Fluid Mechanics": "तरल यांत्रिकी",
    "Kinematics": "गतिकी",
    "Units and Measurements": "मात्रक और मापन",
    "Magnetic Effect of Current": "विद्युत धारा का चुंबकीय प्रभाव",
    "Electromagnetic Induction": "विद्युत चुम्बकीय प्रेरण",
    "AC Circuits": "प्रत्यावर्ती धारा परिपथ",
    "Dual Nature of Matter": "पदार्थ की द्वैत प्रकृति",
    "Communication Systems": "संचार व्यवस्था",
    "Ray Optics": "किरण प्रकाशिकी",
    "Wave Optics": "तरंग प्रकाशिकी",
    "Mole Concept": "मोल संकल्पना",
    "Chemical Equilibrium": "रासायनिक साम्य",
    "Acids Bases and Salts": "अम्ल क्षार और लवण",
    "Electrochemistry": "विद्युत रसायन",
    "Organic Chemistry": "कार्बनिक रसायन",
    "Periodic Table": "आवर्त सारणी",
    "Chemical Bonding": "रासायनिक बंधन",
    "Chemical Kinetics": "रासायनिक बलगतिकी",
    "Order of Reaction": "अभिक्रिया की कोटि",
    "Thermochemistry": "ऊष्मा रसायन",
    "Solutions": "विलयन",
    "Solid State": "ठोस अवस्था",
    "Surface Chemistry": "पृष्ठ रसायन",
    "Coordination Compounds": "उपसहसंयोजक यौगिक",
    "Polymers": "बहुलक",
    "Biomolecules": "जैव अणु",
    "Integration": "समाकलन",
    "Differentiation": "अवकलन",
    "Limits and Continuity": "सीमा और सातत्य",
    "Matrices and Determinants": "आव्यूह और सारणिक",
    "Vectors": "सदिश",
    "Probability": "प्रायिकता",
    "Conic Sections": "शंकु परिच्छेद",
    "Sequences and Series": "अनुक्रम और श्रेणी",
    "Trigonometry": "त्रिकोणमिति",
    "Complex Numbers": "सम्मिश्र संख्याएँ",
    "Sets Relations Functions": "समुच्चय संबंध और फलन",
    "Straight Lines": "सरल रेखाएँ",
    "Binomial Theorem": "द्विपद प्रमेय",
    "Statistics": "सांख्यिकी",
    "Mathematical Reasoning": "गणितीय विवेचन",
    "3D Geometry": "त्रिविमीय ज्यामिति",
    "Differential Equations": "अवकल समीकरण",
    "Relations and Functions": "संबंध और फलन",
    "Inverse Trigonometry": "प्रतिलोम त्रिकोणमिति",
    "Linear Programming": "रैखिक प्रोग्रामन",
    "Cell Biology": "कोशिका जीवविज्ञान",
    "Photosynthesis": "प्रकाश संश्लेषण",
    "Respiration": "श्वसन",
    "Genetics": "आनुवंशिकी",
    "Evolution": "विकास",
    "Ecology": "पारिस्थितिकी",
    "Endocrine System": "अंतःस्रावी तंत्र",
    "Digestive System": "पाचन तंत्र",
    "Nervous System": "तंत्रिका तंत्र",
    "Reproduction": "जनन",
    "Plant Physiology": "पादप शरीर क्रिया विज्ञान",
    "Human Health and Disease": "मानव स्वास्थ्य और रोग",
    "Biotechnology": "जैव प्रौद्योगिकी",
    "Biodiversity": "जैव विविधता",
    "Microbes": "सूक्ष्मजीव",
}


def _get_subtopic_in_language(subtopic_en: str, is_hindi: bool) -> str:
    if not is_hindi:
        return subtopic_en
    return _SUBTOPIC_HINDI_MAP.get(subtopic_en, subtopic_en)


def infer_subtopic_from_question(q: dict, user_subject: str, user_chapter: str,
                                  is_hindi: bool = False) -> str:
    existing = _s(q.get("subtopic"))
    if existing:
        return existing

    question_text = (_s(q.get("question")) + " " + _s(q.get("Explanation"))).lower()
    chapter_lower = user_chapter.lower() if user_chapter else ""
    subject_lower = user_subject.lower() if user_subject else ""

    if "physics" in subject_lower or "phy" in subject_lower or "भौतिक" in subject_lower:
        kw_map = [
            (["friction", "rough", "slipping", "sliding", "coefficient of friction",
              "घर्षण", "खुरदरा", "फिसलन"], "Friction"),
            (["newton", "force", "motion", "inertia", "momentum",
              "न्यूटन", "बल", "गति", "जड़त्व", "संवेग"], "Laws of Motion"),
            (["work", "energy", "power", "kinetic", "potential", "conservative",
              "कार्य", "ऊर्जा", "शक्ति", "गतिज", "स्थितिज"], "Work Energy Power"),
            (["gravitation", "gravity", "orbital", "escape velocity", "satellite",
              "गुरुत्वाकर्षण", "गुरुत्व", "उपग्रह"], "Gravitation"),
            (["current", "resistance", "ohm", "circuit", "kirchhoff", "battery", "cell",
              "धारा", "प्रतिरोध", "परिपथ", "बैटरी"], "Current Electricity"),
            (["electric field", "charge", "coulomb", "potential", "capacitor", "gauss",
              "विद्युत क्षेत्र", "आवेश", "संधारित्र"], "Electrostatics"),
            (["magnetic", "lorentz", "ampere", "solenoid", "biot", "flux",
              "चुम्बकीय", "चुंबक"], "Magnetism"),
            (["wave", "frequency", "amplitude", "interference", "diffraction", "sound",
              "तरंग", "आवृत्ति", "आयाम", "ध्वनि"], "Waves"),
            (["optics", "lens", "mirror", "refraction", "reflection", "prism", "snell",
              "प्रकाशिकी", "लेंस", "दर्पण", "अपवर्तन", "परावर्तन"], "Optics"),
            (["thermodynamics", "heat", "temperature", "entropy", "carnot",
              "ऊष्मागतिकी", "ऊष्मा", "तापमान"], "Thermodynamics"),
            (["semiconductor", "diode", "transistor", "logic gate",
              "अर्धचालक", "डायोड", "ट्रांजिस्टर"], "Semiconductors"),
            (["atom", "bohr", "nuclear", "radioactive", "decay", "fission", "fusion",
              "परमाणु", "नाभिक", "रेडियोसक्रिय"], "Atoms and Nuclei"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    elif "chemistry" in subject_lower or "chem" in subject_lower or "रसायन" in subject_lower:
        kw_map = [
            (["order of reaction", "rate law", "half life", "first order", "zero order",
              "अभिक्रिया की कोटि", "दर नियम", "अर्ध आयु"], "Order of Reaction"),
            (["mole", "avogadro", "stoichiometry", "मोल", "अवोगाद्रो"], "Mole Concept"),
            (["equilibrium", "le chatelier", "साम्य", "ले-शातेलिए"], "Chemical Equilibrium"),
            (["acid", "base", "ph", "buffer", "अम्ल", "क्षार", "बफर"], "Acids Bases and Salts"),
            (["electrochemistry", "galvanic", "electrolysis", "faraday",
              "विद्युत रसायन", "विद्युत अपघटन"], "Electrochemistry"),
            (["organic", "functional group", "iupac", "isomer", "alkane",
              "कार्बनिक", "क्रियात्मक समूह", "आइसोमर"], "Organic Chemistry"),
            (["periodic", "ionization energy", "आवर्त", "आयनन ऊर्जा"], "Periodic Table"),
            (["chemical bonding", "covalent", "ionic", "hybridization",
              "रासायनिक बंध", "सहसंयोजक", "आयनिक"], "Chemical Bonding"),
            (["kinetics", "rate of reaction", "activation energy",
              "अभिक्रिया की दर", "सक्रियण ऊर्जा"], "Chemical Kinetics"),
            (["thermochemistry", "enthalpy", "entropy", "gibbs", "hess",
              "ऊष्मारसायन", "एन्थैल्पी", "एन्ट्रॉपी"], "Thermochemistry"),
            (["solution", "molarity", "molality", "raoult", "osmosis",
              "विलयन", "मोलरता", "परासरण"], "Solutions"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    elif "math" in subject_lower or "maths" in subject_lower or "गणित" in subject_lower:
        kw_map = [
            (["integrate", "integral", "∫", "समाकल", "एकीकरण"], "Integration"),
            (["differentiate", "derivative", "dy/dx", "अवकल", "व्युत्पन्न"], "Differentiation"),
            (["limit", "continuity", "सीमा", "सातत्य"], "Limits and Continuity"),
            (["matrix", "determinant", "आव्यूह", "सारणिक"], "Matrices and Determinants"),
            (["vector", "dot product", "cross product", "सदिश", "अदिश गुणनफल"], "Vectors"),
            (["probability", "bayes", "conditional", "संभावना", "प्रायिकता"], "Probability"),
            (["conic", "parabola", "ellipse", "hyperbola", "circle",
              "शंकु", "परवलय", "दीर्घवृत्त", "अतिपरवलय", "वृत्त"], "Conic Sections"),
            (["sequence", "series", "ap", "gp", "श्रेणी", "अनुक्रम", "समांतर श्रेढ़ी"], "Sequences and Series"),
            (["trigonometry", "sin", "cos", "tan", "त्रिकोणमिति"], "Trigonometry"),
            (["complex number", "argand", "सम्मिश्र संख्या", "आर्गण्ड"], "Complex Numbers"),
            (["set", "relation", "function", "समुच्चय", "संबंध", "फलन"], "Sets Relations Functions"),
            (["straight line", "slope", "सरल रेखा", "ढाल"], "Straight Lines"),
            (["binomial theorem", "द्विपद प्रमेय"], "Binomial Theorem"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    elif "biology" in subject_lower or "bio" in subject_lower or "जीव" in subject_lower:
        kw_map = [
            (["cell", "mitochondria", "nucleus", "कोशिका", "माइटोकॉन्ड्रिया", "केन्द्रक"], "Cell Biology"),
            (["photosynthesis", "chlorophyll", "प्रकाश संश्लेषण", "क्लोरोफिल"], "Photosynthesis"),
            (["respiration", "atp", "glycolysis", "श्वसन", "ग्लाइकोलाइसिस"], "Respiration"),
            (["genetics", "mendel", "allele", "gene", "आनुवंशिकी", "जीन", "मेंडल"], "Genetics"),
            (["evolution", "natural selection", "विकास", "प्राकृतिक चयन"], "Evolution"),
            (["ecology", "ecosystem", "food chain", "पारिस्थितिकी", "पारितंत्र", "खाद्य श्रृंखला"], "Ecology"),
            (["hormone", "endocrine", "हार्मोन", "अंतःस्रावी"], "Endocrine System"),
            (["digestion", "enzyme", "पाचन", "एंजाइम"], "Digestive System"),
            (["nervous", "neuron", "तंत्रिका", "न्यूरॉन"], "Nervous System"),
            (["reproduction", "meiosis", "mitosis", "जनन", "अर्धसूत्री विभाजन"], "Reproduction"),
            (["plant", "root", "stem", "leaf", "पौधा", "जड़", "तना", "पत्ती"], "Plant Physiology"),
        ]
        for keywords, subtopic in kw_map:
            if any(kw in question_text for kw in keywords):
                return _get_subtopic_in_language(subtopic, is_hindi)

    if chapter_lower and len(chapter_lower) > 4:
        return user_chapter.strip()

    return ""


# ================================================================
# OTHER POST-PROCESSING HELPERS
# ================================================================

def scrub_references_from_question(text):
    if not text or not isinstance(text, str):
        return text
    pattern = re.compile(
        r'\s*[-–—]?\s*[\(\[]\s*'
        r'(?:Exercise|Ex\.?|Misc(?:ellaneous)?\s*Exercise|Example|NCERT|PYQ|'
        r'JEE\s*(?:Main|Advanced)?|NEET|AIIMS|CBSE|ICSE|Board)'
        r'[\s\d\.,-]*[\)\]]\s*$',
        re.IGNORECASE
    )
    text = pattern.sub('', text)
    text = re.compile(r'\s*[\(\[]\s*(?:19|20)\d{2}\s*[\)\]]\s*$').sub('', text)
    text = re.sub(r'(\\+\))\)+$', r'\1', text)
    text = re.sub(r'(\.\s*)\)+$', r'\1', text)
    return text.strip()


def is_garbage_field(value, field_name):
    if not value or not isinstance(value, str):
        return False
    garbage_patterns = [
        r'(?i)^(STD\s*\d+\s*[-–]\s*TEST)',
        r'(?i)^TEST\s+(Chapter|Topic|Subtopic|Subject)',
        r'(?i)^(As per image|From image|See image|N/A|null|none)$',
        r'(?i)^(identify from|extract from|identify specific)',
        r'(?i)^(unknown|not specified|not found|not available)$',
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, value.strip()):
            return True
    return False


def text_continues_naturally(prev_text, next_text):
    if not prev_text or not next_text:
        return False
    prev_end = prev_text.strip()[-50:] if len(prev_text) > 50 else prev_text.strip()
    next_start = next_text.strip()[:50] if len(next_text) > 50 else next_text.strip()
    continuation_words = [
        'as', 'the', 'and', 'or', 'but', 'so', 'therefore', 'hence', 'thus',
        'then', 'also', 'if', 'when', 'where', 'which', 'that', 'this', 'these',
        'those', 'its', 'their', 'from', 'with', 'without', 'by', 'for', 'of',
        'to', 'in', 'on', 'at', 'since', 'because', 'while', 'although',
        'और', 'या', 'लेकिन', 'इसलिए', 'अतः', 'यदि', 'जब', 'जहाँ', 'जो',
        'यह', 'वह', 'इस', 'उस', 'से', 'के', 'की', 'को', 'में', 'पर'
    ]
    first_word = next_start.split()[0].lower() if next_start.split() else ""
    ends_mid = prev_end[-1] not in '.!?।'
    is_continuation = (
        ends_mid or
        first_word in continuation_words or
        next_start[0].islower() or
        re.match(r'^[\(\[]', next_start) or
        re.match(r'^\d+[\.\)]', next_start) is None
    )
    return is_continuation


# ================================================================
# SPLIT COMBINED QUESTIONS
# ================================================================

_NEW_Q_AFTER_ANS_RE = re.compile(
    r'(?:Ans(?:wer)?|Sol(?:ution)?|उत्तर|हल)[.\s:→]*'
    r'(?:\(?[A-D1-4]\)?)?\s*'
    r'(?:.*?\n)+?'
    r'\s*(\d+[\.\)]\s+[A-Z\(]|Q\.?\s*\d+[\.\)])',
    re.IGNORECASE | re.DOTALL
)

_ANS_BLOCK_RE = re.compile(
    r'\n?\s*(?:Ans(?:wer)?|Sol(?:ution)?|उत्तर|हल)[.\s:→]*(\(?[A-D1-4]\)?)?\s*',
    re.IGNORECASE
)

_Q_NUMBER_RE = re.compile(
    r'(?:^|\n)\s*(\d+[\.\)]\s+)',
    re.MULTILINE
)

_SUBPART_RE = re.compile(
    r'(?:^|\n)\s*(?:'
    r'[a-d][\.\)]\s+'
    r'|\([a-d]\)\s+'
    r'|\([ivxIVX]+\)\s+'
    r'|[ivx]+[\.\)]\s+'
    r'|\(\d+\)\s+'
    r')',
    re.MULTILINE
)


def split_combined_questions(q):
    question_text = _s(q.get("question"))
    if not question_text:
        return [q]
    if not _NEW_Q_AFTER_ANS_RE.search(question_text):
        return [q]

    if _SUBPART_RE.search(question_text):
        return [q]

    q_starts = [m.start() for m in _Q_NUMBER_RE.finditer(question_text)]
    if len(q_starts) <= 1:
        return [q]
    segments = []
    for i, start in enumerate(q_starts):
        end = q_starts[i + 1] if i + 1 < len(q_starts) else len(question_text)
        segments.append(question_text[start:end].strip())
    if len(segments) <= 1:
        return [q]
    result = []
    for seg in segments:
        new_q = {k: v for k, v in q.items()}
        new_q["questionid"] = ""
        new_q["question"]    = seg
        new_q["Answer"]      = ""
        new_q["Explanation"] = ""
        ans_match = _ANS_BLOCK_RE.search(seg)
        if ans_match:
            new_q["question"] = seg[:ans_match.start()].strip()
            ans_body = seg[ans_match.end():].strip()
            letter_match = re.match(r'^(\(?[A-D1-4]\)?)', ans_body)
            if letter_match:
                raw = re.sub(r'[^A-D1-4]', '', letter_match.group(1)).upper()
                letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
                new_q["Answer"] = letter_map.get(raw, raw)
                ans_body = ans_body[letter_match.end():].strip()
            ans_body = re.sub(
                r'^(?:Sol(?:ution)?|Explanation|व्याख्या|हल)[.\s:→]*',
                '', ans_body, flags=re.IGNORECASE
            ).strip()
            new_q["Explanation"] = ans_body
        if _s(new_q["question"]):
            result.append(new_q)
    return result if len(result) > 1 else [q]


def _s(val):
    return (val or "").strip() if isinstance(val, (str, type(None))) else str(val).strip()


_ANS_PREFIX_RE = re.compile(
    r'^(?:ans(?:wer)?|sol(?:ution)?|therefore|hence|thus|explanation|'
    r'correct\s+(?:option|answer)|the\s+answer|'
    r'उत्तर|हल|अतः|इसलिए|व्याख्या|सही\s+(?:विकल्प|उत्तर))[.\s:→]*',
    re.IGNORECASE | re.UNICODE
)


def is_continuation_fragment(q):
    qid   = _s(q.get("questionid"))
    qtext = _s(q.get("question"))
    if qid == "" and qtext == "":
        return True
    if qid == "" and qtext and _ANS_PREFIX_RE.match(qtext):
        return True
    return False


def _redistribute_fragment_fields(frag):
    frag = dict(frag)
    qtext = _s(frag.get("question"))
    if not qtext:
        return frag
    ans_match = re.match(
        r'^(?:ans(?:wer)?[.\s:→]*|उत्तर[.\s:→]*)\(?([A-D1-4])\)?\s*[,.]?\s*',
        qtext, re.IGNORECASE | re.UNICODE
    )
    if ans_match:
        frag["Answer"] = frag.get("Answer") or ans_match.group(1)
        rest = qtext[ans_match.end():].strip()
        rest = re.sub(r'^(?:sol(?:ution)?|explanation|हल|व्याख्या)[.\s:→]*', '', rest,
                      flags=re.IGNORECASE).strip()
        if rest:
            frag["Explanation"] = ((_s(frag.get("Explanation")) + " " + rest).strip()
                                   if _s(frag.get("Explanation")) else rest)
        frag["question"] = ""
        return frag
    if _ANS_PREFIX_RE.match(qtext):
        text = _ANS_PREFIX_RE.sub("", qtext).strip()
        if text:
            frag["Explanation"] = ((_s(frag.get("Explanation")) + " " + text).strip()
                                   if _s(frag.get("Explanation")) else text)
        frag["question"] = ""
    return frag


def _question_needs_answer(q):
    return _s(q.get("Answer")) == "" and _s(q.get("Explanation")) == ""


def merge_question_parts(base_q, continuation_q):
    continuation_q = _redistribute_fragment_fields(continuation_q)
    merged = dict(base_q)
    base_question = _s(merged.get("question"))
    cont_question = _s(continuation_q.get("question"))
    if cont_question:
        merged["question"] = (base_question + " " + cont_question).strip() if base_question else cont_question
    base_exp = _s(merged.get("Explanation"))
    cont_exp = _s(continuation_q.get("Explanation"))
    if cont_exp:
        merged["Explanation"] = (base_exp + " " + cont_exp).strip() if base_exp else cont_exp
    for i in range(1, 5):
        key = f"option{i}"
        if not _s(merged.get(key)) and _s(continuation_q.get(key)):
            merged[key] = continuation_q[key]
    if not _s(merged.get("Answer")) and _s(continuation_q.get("Answer")):
        merged["Answer"] = continuation_q["Answer"]
    if not _s(str(merged.get("marks", ""))) and _s(str(continuation_q.get("marks", ""))):
        merged["marks"] = continuation_q["marks"]
    if not _s(merged.get("previous_year")) and _s(continuation_q.get("previous_year")):
        merged["previous_year"] = continuation_q["previous_year"]
    if not _s(merged.get("subtopic")) and _s(continuation_q.get("subtopic")):
        merged["subtopic"] = continuation_q["subtopic"]
    if not _s(merged.get("question_bucket")) and _s(continuation_q.get("question_bucket")):
        merged["question_bucket"] = continuation_q["question_bucket"]
    return merged


# ================================================================
# MARKS / ANSWER / FIELD FIXES
# ================================================================
_SECTION_EACH_RE = re.compile(
    r'(\d+)\s*[Mm]arks?\s+[Ee]ach|'
    r'[Ee]ach\s+(?:of\s+)?(\d+)\s*[Mm]arks?|'
    r'questions?\s+of\s+(\d+)\s*[Mm]arks?|'
    r'(\d+)\s*अंक\s+(?:प्रत्येक|each)|'
    r'प्रत्येक\s+(?:प्रश्न\s+)?(\d+)\s*अंक',
)


def extract_section_marks_from_text(text):
    if not text:
        return ""
    m = _SECTION_EACH_RE.search(str(text))
    if m:
        for g in m.groups():
            if g:
                return g
    return ""


def fix_marks_field(marks_val):
    if marks_val is None:
        return ""
    val = str(marks_val).strip()
    if re.match(r'^\d+$', val):
        return val
    patterns = [
        r'\[(\d+)\s*[Mm]arks?\]', r'\((\d+)\s*[Mm]arks?\)',
        r'[Mm]arks?\s*[:\-]\s*(\d+)', r'(\d+)\s*[Mm]arks?',
        r'\[(\d+)\s*अंक\]', r'\((\d+)\s*अंक\)',
        r'(\d+)\s*अंक',
        r'\[(\d+)\]', r'\((\d+)\)', r'(\d+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, val)
        if m:
            return m.group(1)
    return ""


def infer_marks_from_sections(all_questions, page_marks_map=None, question_page_map=None):
    if page_marks_map and question_page_map:
        for i, q in enumerate(all_questions):
            page_idx = question_page_map.get(i, -1)
            if page_idx >= 0 and page_idx < len(page_marks_map):
                section_mark = page_marks_map[page_idx]
                if section_mark:
                    existing = fix_marks_field(q.get("marks", ""))
                    if not existing:
                        q["marks"] = section_mark

    current_section_marks = ""
    for q in all_questions:
        candidate_texts = [q.get("marks", ""), q.get("Explanation", ""), q.get("question", "")]
        found_marks = ""
        for text in candidate_texts:
            found_marks = extract_section_marks_from_text(text)
            if found_marks:
                break
        if found_marks:
            current_section_marks = found_marks
        marks_now = str(q.get("marks", "")).strip()
        cleaned_marks = fix_marks_field(marks_now)
        if cleaned_marks:
            q["marks"] = cleaned_marks
        elif current_section_marks:
            q["marks"] = current_section_marks

    for i in range(len(all_questions) - 1, -1, -1):
        if not str(all_questions[i].get("marks", "")).strip():
            for j in range(i + 1, min(i + 10, len(all_questions))):
                fwd = str(all_questions[j].get("marks", "")).strip()
                if fwd:
                    all_questions[i]["marks"] = fwd
                    break

    return all_questions


def fix_answer_field(answer_val, options, question_type):
    q_type = str(question_type).lower()
    if "true" in q_type or "false" in q_type or "सत्य" in q_type or "असत्य" in q_type:
        if not answer_val or not isinstance(answer_val, str):
            return ""
        val = answer_val.strip().lower()
        if val in ("true", "1", "t", "yes", "सत्य", "हाँ"):
            return "1"
        if val in ("false", "0", "f", "no", "असत्य", "नहीं"):
            return "0"
        return answer_val
    if "numeric" in q_type or "integer" in q_type or "संख्यात्मक" in q_type:
        if not answer_val and answer_val != 0:
            return ""
        val = str(answer_val).strip()
        try:
            return str(int(float(val)))
        except Exception:
            m = re.search(r'\d+', val)
            return m.group(0) if m else val
    if "mcq" in q_type or "multiple" in q_type or "बहुविकल्पीय" in q_type:
        if not answer_val or not isinstance(answer_val, str):
            return ""
        val = answer_val.strip()
        if val in ('1', '2', '3', '4'):
            return val
        letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4',
                      'अ': '1', 'ब': '2', 'स': '3', 'द': '4'}
        inner = re.sub(r'[()]', '', val).strip().upper()
        if inner in letter_map:
            return letter_map[inner]
        if inner in ('1', '2', '3', '4'):
            return inner
        return answer_val
    return str(answer_val).strip() if answer_val else ""


# ================================================================
# QUESTION BUCKET NORMALISER
# ================================================================
VALID_BUCKETS = {"Beginner", "Target", "Advance Climb", "Must Do"}


def normalize_question_bucket(bucket_raw):
    if not bucket_raw or not isinstance(bucket_raw, str):
        return "Beginner"
    val = bucket_raw.strip()
    if val in VALID_BUCKETS:
        return val
    lower = val.lower()
    for valid in VALID_BUCKETS:
        if valid.lower() == lower:
            return valid
    if "advance" in lower or "climb" in lower:
        return "Advance Climb"
    if "must" in lower or "critical" in lower or "important" in lower:
        return "Must Do"
    if "target" in lower or "moderate" in lower or "medium" in lower:
        return "Target"
    if "begin" in lower or "easy" in lower or "basic" in lower or "simple" in lower:
        return "Beginner"
    return "Beginner"


# ================================================================
# QUESTION TYPE NORMALISER
# ================================================================
def normalize_question_type(q_type_raw):
    q_type = str(q_type_raw).strip()
    lower  = q_type.lower()
    if "true" in lower or "false" in lower or "सत्य" in lower or "असत्य" in lower:
        return "True/False"
    elif "assertion" in lower or "कथन" in lower:
        return "Assertion and Reasoning Questions ( A& R )"
    elif "match" in lower or "मिलान" in lower:
        return "Match the Column Question"
    elif "case" in lower or "प्रकरण" in lower:
        return "Case Based Questions (CBQ)"
    elif "blank" in lower or "filling" in lower or "fill" in lower or "रिक्त" in lower:
        return "Filling Blank"
    elif "numeric" in lower or "integer" in lower or "संख्यात्मक" in lower:
        return "Numeric"
    elif "subjective" in lower or "दीर्घ" in lower or "लघु" in lower:
        return "Subjective"
    else:
        return "MCQs"


def enforce_question_type_rules(q):
    q_type = q.get("question_type", "MCQs")

    if q_type == "MCQs":
        ans = str(q.get("Answer", "")).strip()
        if ans not in ("1", "2", "3", "4"):
            letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
            inner = re.sub(r'[^A-Da-d1-4]', '', ans).upper()
            if inner and inner[0] in letter_map:
                q["Answer"] = letter_map[inner[0]]
            elif inner and inner[0] in ('1', '2', '3', '4'):
                q["Answer"] = inner[0]
            else:
                q["Answer"] = ""

    elif q_type == "True/False":
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""
        ans = str(q.get("Answer", "")).strip().lower()
        if ans in ("true", "1", "t", "yes", "सत्य") or (ans and "true" in ans and "false" not in ans):
            q["Answer"] = "1"
        elif ans in ("false", "0", "f", "no", "असत्य") or (ans and "false" in ans):
            q["Answer"] = "0"
        else:
            q["Answer"] = ""

    elif q_type == "Numeric":
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""
        ans = str(q.get("Answer", "")).strip()
        try:
            q["Answer"] = str(int(float(ans))) if ans else ""
        except Exception:
            m = re.search(r'\d+', ans)
            q["Answer"] = m.group(0) if m else ""

    elif q_type == "Filling Blank":
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""

    elif q_type in (
        "Subjective",
        "Assertion and Reasoning Questions ( A& R )",
        "Match the Column Question",
        "Case Based Questions (CBQ)",
    ):
        q["option1"] = q["option2"] = q["option3"] = q["option4"] = ""
        q["Answer"] = ""

    return q


def apply_field_order(q):
    ordered = {}
    for k in FIELD_ORDER:
        ordered[k] = q.get(k, "")
    for k, v in q.items():
        if k not in ordered and k != "category":
            ordered[k] = v
    return ordered


_ANY_TABLE_RE = re.compile(
    r'(<table\b[^>]*>.*?</table\s*>)',
    re.IGNORECASE | re.DOTALL
)

_SOLUTION_SNEAK_RE = re.compile(
    r'(?:<br\s*/?>|\n|\s{2,})\s*'
    r'(?:'
    r'sol(?:ution)?\.?\s*[:\-→]?|'
    r'ans(?:wer)?\.?\s*[:\-→]?|'
    r'explanation\s*[:\-→]?|'
    r'lcm\s*[=:]|hcf\s*[=:]|'
    r'∴\s*lcm|∴\s*hcf|'
    r'therefore[,\s]|hence[,\s]|thus[,\s]|'
    r'उत्तर\s*[:\-]?|हल\s*[:\-]?|व्याख्या\s*[:\-]?'
    r')',
    re.IGNORECASE | re.DOTALL
)


def fix_misplaced_tables(q):
    q_type = q.get("question_type", "")
    if "Match" in q_type:
        return q

    question_text = q.get("question", "")
    if not question_text or not isinstance(question_text, str):
        return q

    extra_parts = []
    clean_q = question_text

    if '<table' in clean_q.lower():
        tables_found = _ANY_TABLE_RE.findall(clean_q)
        if tables_found:
            clean_q = _ANY_TABLE_RE.sub('', clean_q)
            extra_parts.extend(tables_found)

    _ladder_in_q = re.search(r'(?:<br>|\n|\s{2,})\s*\d+\s*\|', clean_q)
    if _ladder_in_q and _ladder_in_q.start() > 10:
        ladder_tail = clean_q[_ladder_in_q.start():]
        clean_q = clean_q[:_ladder_in_q.start()].strip()
        if ladder_tail.strip():
            extra_parts.append(ladder_tail.strip())

    sol_match = _SOLUTION_SNEAK_RE.search(clean_q)
    if sol_match and sol_match.start() > 15:
        solution_tail = clean_q[sol_match.start():].strip()
        clean_q = clean_q[:sol_match.start()].strip()
        if solution_tail:
            extra_parts.insert(0, solution_tail)

    if not extra_parts:
        return q

    clean_q = re.sub(r'(<br\s*/?>\s*){2,}', '<br>', clean_q).strip()
    clean_q = re.sub(r'\s{2,}', ' ', clean_q).strip()

    existing_exp = (q.get("Explanation", "") or "").strip()
    rescued = '\n'.join(p.strip() for p in extra_parts if p.strip())
    q["Explanation"] = (rescued + '\n' + existing_exp).strip() if existing_exp else rescued
    q["question"] = clean_q
    return q


# ================================================================
# UNIFIED CLEAN_QUESTION
# ================================================================
def clean_question(q, page_images=None,
                   user_subject="", user_course="", user_class="",
                   user_chapter="", user_practice="", user_book="",
                   is_hindi=False):
    math_fields     = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    metadata_fields = ["subjectname", "chapter", "practice", "subtopic", "medium",
                       "difficulty", "question_type", "course"]

    for k in list(q.keys()):
        if q[k] is None:
            q[k] = ""

    q.pop("category", None)
    q["question_type"] = normalize_question_type(q.get("question_type") or "MCQs")
    q["question_bucket"] = normalize_question_bucket(q.get("question_bucket") or "")

    q["medium"] = "Hindi" if is_hindi else "English"

    existing_py = fix_previous_year_field(_s(q.get("previous_year")))
    if not existing_py:
        q_text = _s(q.get("question"))
        ref, cleaned_q = extract_previous_year_from_text(q_text)
        if ref:
            existing_py = fix_previous_year_field(ref)
            if existing_py:
                q["question"] = cleaned_q
        if not existing_py:
            exp_text = _s(q.get("Explanation"))
            ref, cleaned_exp = extract_previous_year_from_text(exp_text)
            if ref:
                existing_py = fix_previous_year_field(ref)
                if existing_py:
                    q["Explanation"] = cleaned_exp

    q["previous_year"] = existing_py

    if "question" in q:
        q["question"] = scrub_references_from_question(q["question"])

    for field in list(q.keys()):
        if isinstance(q[field], str):
            q[field] = remove_cite_tags(q[field])

    if "Explanation" in q:
        q["Explanation"] = clean_explanation_prefix(q["Explanation"])
        if q["Explanation"] and isinstance(q["Explanation"], str):
            q["Explanation"] = re.sub(
                r'<(?!img\b|/img\b|table\b|/table\b|tr\b|/tr\b|td\b|/td\b|th\b|/th\b|br\b|div\b|/div\b)[^>]+>',
                '', q["Explanation"]
            )

    for field in math_fields:
        if field in q:
            q[field] = convert_dollar_to_latex(q[field])

    if "Answer" in q:
        options = {
            "A": q.get("option1") or "", "B": q.get("option2") or "",
            "C": q.get("option3") or "", "D": q.get("option4") or "",
        }
        q["Answer"] = fix_answer_field(q["Answer"], options, q["question_type"])

    q["marks"] = fix_marks_field(q.get("marks") or "")

    for field in metadata_fields:
        if field in q and is_garbage_field(q[field], field):
            q[field] = ""

    q = enforce_question_type_rules(q)

    q = inject_diagrams_into_question(q, page_images or [])

    if user_subject:  q["subjectname"] = user_subject
    if user_course:   q["course"]      = user_course
    if user_class:    q["class"]       = user_class
    if user_chapter:  q["chapter"]     = user_chapter
    q["practice"] = user_practice if user_practice else q.get("practice", "")
    if user_book:     q["book"]        = user_book

    q["medium"] = "Hindi" if is_hindi else "English"

    if not _s(q.get("subtopic")):
        inferred = infer_subtopic_from_question(q, user_subject, user_chapter, is_hindi=is_hindi)
        if inferred:
            q["subtopic"] = inferred

    q = fix_misplaced_tables(q)

    br_fields = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    for field in br_fields:
        if field in q:
            q[field] = newline_to_br(q[field])

    return apply_field_order(q)


# ================================================================
# JSON CLEANER
# ================================================================
def _syntax_repair(text):
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


def _close_open_structures(text):
    text = text.rstrip().rstrip(',')
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket -= 1
    text += '}' * max(0, depth_brace)
    text += ']' * max(0, depth_bracket)
    return text


def clean_json_response(text):
    if not text:
        return None
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    def _try_parse(s):
        s = _syntax_repair(s)
        try:
            return json.loads(s)
        except Exception:
            pass
        s2 = _close_open_structures(s)
        try:
            result = json.loads(s2)
            if result:
                print(f"⚠️ JSON repaired: {len(result) if isinstance(result, list) else 1} item(s)")
            return result
        except Exception:
            return None

    def _only_dicts(lst):
        if not isinstance(lst, list):
            return None
        dicts = [x for x in lst if isinstance(x, dict)]
        return dicts if dicts else None

    start_idx = text.find('[')
    end_idx   = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        result = _try_parse(text[start_idx:end_idx + 1])
        if result and isinstance(result, list):
            clean = _only_dicts(result)
            if clean:
                return clean
        result = _try_parse(text[start_idx:])
        if result and isinstance(result, list):
            clean = _only_dicts(result)
            if clean:
                return clean

    obj_start = text.find('{')
    obj_end   = text.rfind('}')
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        result = _try_parse(text[obj_start:obj_end + 1])
        if result and isinstance(result, dict):
            return [result]

    objects = []
    depth = 0
    in_string = False
    escape = False
    obj_start_pos = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                obj_start_pos = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start_pos is not None:
                fragment = text[obj_start_pos:i + 1]
                try:
                    obj = json.loads(_syntax_repair(fragment))
                    if isinstance(obj, dict) and obj:
                        objects.append(obj)
                except Exception:
                    pass
                obj_start_pos = None
    if objects:
        print(f"⚠️ JSON recovered {len(objects)} object(s)")
        return objects

    print("⚠️ JSON Parse failed")
    return None


# ================================================================
# ENHANCED STITCHING FUNCTION
# ================================================================

def stitch_split_questions_enhanced(page_results):
    num_pages = len(page_results)

    for _ in range(15):
        merged_any = False

        for page_idx in range(num_pages):
            page = page_results[page_idx]
            if not page:
                continue
            i = 1
            while i < len(page):
                if is_continuation_fragment(page[i]):
                    page[i - 1] = merge_question_parts(page[i - 1], page[i])
                    page.pop(i)
                    merged_any = True
                else:
                    i += 1
            page_results[page_idx] = page

        for page_idx in range(num_pages - 1):
            # ── FIX 2: guard empty page list before accessing [-1] or [0] ──
            if not page_results[page_idx]:
                continue
            next_idx = page_idx + 1
            while next_idx < num_pages and not page_results[next_idx]:
                next_idx += 1
            if next_idx >= num_pages:
                continue

            # ── FIX 2: double-check both lists are non-empty ──
            if not page_results[page_idx] or not page_results[next_idx]:
                continue

            last_q  = page_results[page_idx][-1]
            first_q = page_results[next_idx][0]

            should_merge = False

            if is_continuation_fragment(first_q):
                should_merge = True
            elif _question_needs_answer(last_q):
                first_qid  = _s(first_q.get("questionid"))
                first_ans  = _s(first_q.get("Answer"))
                first_exp  = _s(first_q.get("Explanation"))
                first_qtxt = _s(first_q.get("question"))
                can_supply = (
                    first_qid == "" or
                    first_ans != "" or
                    first_exp != "" or
                    bool(_ANS_PREFIX_RE.match(first_qtxt))
                )
                if can_supply:
                    should_merge = True
            elif (
                _s(last_q.get("questionid")) != "" and
                _s(first_q.get("questionid")) == _s(last_q.get("questionid")) and
                (
                    _question_needs_answer(last_q) or
                    not _s(last_q.get("option1")) or
                    not _s(last_q.get("option2"))
                )
            ):
                should_merge = True

            if should_merge:
                page_results[page_idx][-1] = merge_question_parts(last_q, first_q)
                page_results[next_idx].pop(0)
                merged_any = True
                break

        if not merged_any:
            break

    for page_idx in range(num_pages):
        if page_results[page_idx]:
            page_results[page_idx] = [
                q for q in page_results[page_idx]
                if any(_s(q.get(f)) for f in ("question", "Explanation", "Answer",
                                               "option1", "option2", "option3", "option4"))
            ]

    return page_results


# ================================================================
# TARGETED CONTINUATION EXTRACTOR
# ================================================================
def extract_continuation_fragment(pdf_path, page_index, prev_last_q, api_key, model_name,
                                  user_subject, user_course, user_class,
                                  user_chapter, user_practice, user_book,
                                  skip_hashes=None, is_hindi=False):
    try:
        client = genai.Client(api_key=api_key)
        img_bytes = pdf_page_to_png_bytes(pdf_path, page_index, dpi=200)
        page_images = extract_all_page_images(pdf_path, page_index, skip_hashes=skip_hashes)

        prev_q_text = str(prev_last_q.get("question", ""))[:500]
        prev_q_expl = str(prev_last_q.get("Explanation", ""))[:300]

        lang_instruction = (
            "LANGUAGE: This is a Hindi medium PDF. Extract ALL text in Hindi exactly as it appears. "
            "Do NOT translate to English. Preserve all Devanagari script.\n\n"
            if is_hindi else ""
        )

        medium_val = "Hindi" if is_hindi else "English"

        prompt = f"""
You are recovering a MISSING answer/solution for an incomplete question.

{lang_instruction}

INCOMPLETE QUESTION (from previous page):
"{prev_q_text}"
{f'Partial solution so far: "{prev_q_expl}"' if prev_q_expl else '(No solution extracted yet)'}

YOUR TASK: Scan this page image carefully. Find the answer and/or solution that belongs to the above question.

STRICT RULES:
1. Copy text EXACTLY as it appears in the image — word for word, character for character.
2. Do NOT rephrase, summarize, shorten, or add anything of your own.
3. The answer/solution will appear BEFORE the first new numbered question on this page.
4. Extract EVERY line of the solution — do not skip any steps.
5. Convert all math to LaTeX: inline \\\\( ... \\\\), display \\\\[ ... \\\\]
6. If answer letter found (A/B/C/D), map: A→1, B→2, C→3, D→4
7. IMPORTANT — previous_year: Look for references like (Example-1), (Exercise-7.1-8), JEE Main 2024,CBSE 2025, CBSE, NCERT etc.
8. LCM/HCF LADDER RULE (most critical):
   If the solution has a prime factorisation / division ladder, write it as PLAIN TEXT:
   2 | 15  20  25
     |___________
   2 | 15  10  25
     |___________
   3 | 15   5  25
     |___________
   5 |  5   5  25
     |___________
   5 |  1   1   5
     |___________
       1   1   1
   Rules: "divisor | numbers" for each step, "  |___" separator after each row,
   "    1  1  1" (indented, no pipe) for the final row. Use PLAIN TEXT — NO HTML table.
   For OTHER tables (data/frequency/match-column): use HTML <table class="ltable" style="border-collapse:collapse"> with <td style="border:1px solid #333;padding:5px 10px">.

Return ONLY this JSON:
{{
  "Answer": "<1/2/3/4 or 1/0 for True/False or numeric, or empty if not found>",
  "Explanation": "<exact verbatim solution text from the image, every step — tables as HTML>",
  "question": "<verbatim continuation of question text if it continues here, else empty>",
  "option1": "<verbatim continuation of option A if split, else empty>",
  "option2": "<verbatim continuation of option B if split, else empty>",
  "option3": "<verbatim continuation of option C if split, else empty>",
  "option4": "<verbatim continuation of option D if split, else empty>",
  "previous_year": "<extracted reference or empty>"
}}

If nothing related to this question appears on this page, return: {{}}
"""

        for attempt, cfg in enumerate(_make_page_configs()[:3]):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, genai_types.Part.from_bytes(data=img_bytes, mime_type='image/png')],
                    config=cfg,
                )
                raw = _safe_response_text(response)
                if not raw:
                    continue
                text = raw.strip()
                text = re.sub(r'```json\s*', '', text)
                text = re.sub(r'```\s*', '', text)

                result = json.loads(text)
                if result and (result.get("question") or result.get("Explanation") or result.get("Answer")):
                    fragment = {
                        "questionid": "",
                        "question": result.get("question", ""),
                        "option1": result.get("option1", ""),
                        "option2": result.get("option2", ""),
                        "option3": result.get("option3", ""),
                        "option4": result.get("option4", ""),
                        "Answer": result.get("Answer", ""),
                        "Explanation": result.get("Explanation", ""),
                        "course": "", "subjectname": "", "chapter": "", "practice": "",
                        "subtopic": "", "medium": medium_val,
                        "difficulty": "", "question_type": "",
                        "previous_year": result.get("previous_year", ""),
                        "marks": "", "class": "", "book": "", "question_bucket": "",
                        "diagram_placements": []
                    }

                    fragment = clean_question(
                        fragment, page_images=page_images,
                        user_subject=user_subject, user_course=user_course,
                        user_class=user_class, user_chapter=user_chapter,
                        user_practice=user_practice, user_book=user_book,
                        is_hindi=is_hindi
                    )
                    return fragment
                return None
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                continue
        return None
    except Exception:
        return None


# ================================================================
# TARGETED ANSWER RECOVERY
# ================================================================
def recover_missing_answers(pdf_path, page_results, api_key, model_name,
                            user_subject, user_course, user_class,
                            user_chapter, user_practice, user_book, status_placeholder,
                            skip_hashes=None, is_hindi=False):
    num_pages = len(page_results)
    recovered = 0

    targets = []
    for page_idx in range(num_pages):
        page = page_results[page_idx]
        # ── FIX 2: also guard empty list here ──
        if not page or not _question_needs_answer(page[-1]):
            continue
        candidates = []
        nxt = page_idx + 1
        while nxt < num_pages and len(candidates) < 2:
            candidates.append(nxt)
            nxt += 1
        if candidates:
            targets.append((page_idx, candidates))

    if not targets:
        return page_results, 0

    status_placeholder.update(
        label=f"Answer Recovery: {len(targets)} incomplete question(s) — fixing..."
    )

    for page_idx, candidate_pages in targets:
        # ── FIX 2: guard again before accessing [-1] ──
        if not page_results[page_idx]:
            continue
        current_q = page_results[page_idx][-1]
        found = False

        for next_idx in candidate_pages:
            if found or not page_results[next_idx]:
                continue
            first_q = page_results[next_idx][0]
            first_ans  = _s(first_q.get("Answer"))
            first_exp  = _s(first_q.get("Explanation"))
            first_qtxt = _s(first_q.get("question"))
            first_qid  = _s(first_q.get("questionid"))

            can_supply = (
                first_qid == "" or
                first_ans != "" or
                first_exp != "" or
                bool(_ANS_PREFIX_RE.match(first_qtxt))
            )
            if can_supply:
                page_results[page_idx][-1] = merge_question_parts(current_q, first_q)
                page_results[next_idx].pop(0)
                current_q = page_results[page_idx][-1]
                recovered += 1
                found = True

        if not found:
            for next_idx in candidate_pages:
                if found:
                    break
                for attempt in range(3):
                    try:
                        fragment = extract_continuation_fragment(
                            pdf_path, next_idx, current_q, api_key, model_name,
                            user_subject, user_course, user_class,
                            user_chapter, user_practice, user_book,
                            skip_hashes=skip_hashes, is_hindi=is_hindi
                        )
                        if fragment and (_s(fragment.get("Answer")) or _s(fragment.get("Explanation"))):
                            page_results[page_idx][-1] = merge_question_parts(current_q, fragment)
                            current_q = page_results[page_idx][-1]
                            recovered += 1
                            found = True
                            break
                        time.sleep(1)
                    except Exception:
                        time.sleep(2)

    if recovered:
        page_results = stitch_split_questions_enhanced(page_results)

    return page_results, recovered


# ================================================================
# FINAL CLEANUP
# ================================================================
def final_cleanup(all_questions):
    seen_signatures = set()
    cleaned = []
    duplicates_removed = 0

    for q in all_questions:
        question_text = _s(q.get("question"))
        if len(question_text) < 4:
            continue
        opts = "".join([_s(q.get(f"option{i}")) for i in range(1, 5)])
        signature = (question_text + opts).lower().replace(" ", "")
        if signature in seen_signatures:
            duplicates_removed += 1
            continue
        seen_signatures.add(signature)
        cleaned.append(q)

    return cleaned, duplicates_removed


def fix_exam_field(all_questions):
    first_exam = None
    for q in all_questions:
        exam = q.get("course") or q.get("EXAM")
        if exam and exam not in ("null", "", None):
            first_exam = exam
            break
    if first_exam:
        for q in all_questions:
            if not q.get("course"):
                q["course"] = first_exam
    return all_questions


# ================================================================
# DEDUPLICATE DIAGRAM PLACEMENTS
# ================================================================
def deduplicate_diagram_placements(questions: list) -> list:
    if not questions:
        return questions

    for q in questions:
        for field in _IMG_FIELDS:
            if field in q and q[field]:
                q[field] = _normalize_diagram_refs(q[field])

    diagram_to_qs: dict[int, list[int]] = {}
    for qi, q in enumerate(questions):
        for field in _IMG_FIELDS:
            for m in re.finditer(r'\[DIAGRAM_(\d+)\]', q.get(field, "") or ""):
                idx = int(m.group(1))
                diagram_to_qs.setdefault(idx, [])
                if qi not in diagram_to_qs[idx]:
                    diagram_to_qs[idx].append(qi)

    _VISUAL_KWS = [
        'figure', 'fig', 'diagram', 'triangle', 'circle', 'rectangle',
        'graph', 'circuit', 'image', 'refer', 'shown', 'given', 'above',
        'shape', 'polygon', 'coordinate', 'axis', 'chart', 'map',
        'चित्र', 'आकृति', 'दिया', 'ग्राफ', 'आरेख',
    ]

    for idx, qi_list in diagram_to_qs.items():
        if len(qi_list) <= 1:
            continue

        tag = f'[DIAGRAM_{idx}]'

        def _score(qi):
            q = questions[qi]
            combined = " ".join([
                (q.get("question", "") or ""),
                (q.get("Explanation", "") or ""),
            ]).lower()
            return sum(1 for kw in _VISUAL_KWS if kw in combined)

        best_qi = max(qi_list, key=_score)

        for qi in qi_list:
            if qi == best_qi:
                continue
            q = questions[qi]
            for field in _IMG_FIELDS:
                if field in q and q[field] and tag in q[field]:
                    q[field] = q[field].replace(tag, "").strip()

    return questions


# ================================================================
# AUTO-INJECT MISSED DIAGRAMS
# ================================================================
def auto_inject_missed_diagrams(questions, page_images):
    if not questions or not page_images:
        return questions

    for q in questions:
        for field in _IMG_FIELDS:
            if field in q and q[field]:
                q[field] = _normalize_diagram_refs(q[field])

    placed: set[int] = set()
    for q in questions:
        for field in _IMG_FIELDS:
            for m in re.finditer(r'\[DIAGRAM_(\d+)\]', q.get(field, "") or ""):
                placed.add(int(m.group(1)))

    missing = [i for i in range(len(page_images)) if i not in placed]
    if not missing:
        return questions

    _STRONG_KWS = [
        'figure', 'fig.', 'fig ', 'diagram', 'refer', 'shown', 'given below',
        'given above', 'graph', 'circuit', 'image', 'picture', 'illustration',
        'chart', 'map', 'plot', 'shape', 'triangle', 'circle', 'rectangle',
        'polygon', 'trapezium', 'rhombus', 'coordinate', 'axis',
        'चित्र', 'आकृति', 'दिया गया', 'नीचे दिए', 'ऊपर दिए', 'आरेख', 'ग्राफ',
    ]

    for idx in missing:
        tag = f'[DIAGRAM_{idx}]'
        for q in questions:
            q_text = (q.get("question", "") or "").lower()
            exp_text = (q.get("Explanation", "") or "").lower()
            combined = q_text + " " + exp_text
            if any(kw in combined for kw in _STRONG_KWS):
                already_has = any('[DIAGRAM_' in (q.get(f, "") or "") for f in _IMG_FIELDS)
                if not already_has:
                    q["question"] = (q.get("question", "") or "") + f"\n{tag}"
                    break

    return questions


# ================================================================
# ENHANCED GEMINI PAGE PROCESSOR
# ================================================================
def process_single_page(args):
    (pdf_path, page_index, page_num, api_key, model_name,
     user_subject, user_course, user_class, user_chapter, user_practice,
     user_book, section_marks_hint, skip_hashes, is_hindi) = args
    try:
        client = genai.Client(api_key=api_key)
        img_bytes = pdf_page_to_png_bytes(pdf_path, page_index, dpi=250)
        page_images = extract_all_page_images(pdf_path, page_index, skip_hashes=skip_hashes)
        num_diagrams = len(page_images)

        if num_diagrams > 0:
            diagram_note = (
                f'+=================================================================+\n'
                f'|  DIAGRAM RULE — {num_diagrams} IMAGE(S) — READ EVERY POINT CAREFULLY  |\n'
                f'+=================================================================+\n\n'
                f'I am sending you {num_diagrams} extracted image(s) AFTER the full page image.\n'
                f'They are ordered TOP-TO-BOTTOM as they appear on the page:\n'
                f'  [DIAGRAM_0] = topmost image on page\n'
                + (f'  [DIAGRAM_1] = second image from top\n' if num_diagrams > 1 else '')
                + (f'  ... up to [DIAGRAM_{num_diagrams-1}] = bottommost\n' if num_diagrams > 2 else '')
                + f'\n'
                f'SEQUENCE RULE (MOST IMPORTANT):\n'
                f'  Read the page TOP to BOTTOM. Place each [DIAGRAM_X] AT THE EXACT POINT\n'
                f'  in the text where that image physically appears on the page.\n'
                f'  The order of [DIAGRAM_X] tags in your output MUST match reading order.\n\n'
                f'PLACEMENT RULES:\n'
                f'  • Image appears ABOVE question text          → [DIAGRAM_X] at very START of "question"\n'
                f'  • Image appears IN MIDDLE of question text   → [DIAGRAM_X] inserted mid-sentence\n'
                f'  • Image appears AFTER question, BEFORE opts  → [DIAGRAM_X] at END of "question"\n'
                f'  • Image appears INSIDE an option             → [DIAGRAM_X] inside that option field\n'
                f'  • Image appears in solution/working          → [DIAGRAM_X] inside "Explanation"\n\n'
                f'EXAMPLES of correct in-sequence placement:\n'
                f'  "question": "In the figure [DIAGRAM_0] find the perimeter of triangle ABC."\n'
                f'  "question": "[DIAGRAM_0] The triangle shown above has sides AB=5, BC=12."\n'
                f'  "Explanation": "From the graph [DIAGRAM_1], mode = 25, mean = 22."\n\n'
                f'STRICT RULES:\n'
                f'  • Use EVERY index 0–{num_diagrams-1} exactly once.\n'
                f'  • NEVER skip a diagram. NEVER describe it in words instead of placing [DIAGRAM_X].\n'
                f'  • NEVER move a diagram to a different question than where it appears.'
            )
        else:
            diagram_note = (
                'No diagrams were detected on this page.\n'
                'If you see any figure, graph, or shape → place [DIAGRAM_0] at that position.'
            )

        marks_hint_text = ""
        if section_marks_hint:
            marks_hint_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION MARKS: ALL questions on this page carry {section_marks_hint} mark(s).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        subject_ctx = f'Subject: "{user_subject}"' if user_subject else ""
        chapter_ctx = f'Chapter: "{user_chapter}"' if user_chapter else ""
        practice_ctx = f'Practice/Exercise: "{user_practice}"' if user_practice else ""
        context_block = " | ".join(filter(None, [subject_ctx, chapter_ctx, practice_ctx]))

        medium_val = "Hindi" if is_hindi else "English"

        if is_hindi:
            language_block = """
══════════════════════════════════════════════
CRITICAL LANGUAGE RULE — HINDI MEDIUM PDF
══════════════════════════════════════════════
This PDF is in HINDI (Devanagari script).

MANDATORY RULES:
1. Extract ALL text EXACTLY as it appears in Hindi — do NOT translate to English.
2. Copy every word in Devanagari script verbatim.
3. Questions, options, explanations — ALL must be in Hindi as printed.
4. Math formulas/numbers can remain as-is (LaTeX for equations).
5. Field values like "question", "option1"–"option4", "Explanation" MUST be in Hindi.
6. "subtopic" MUST also be in Hindi (e.g. "अभिक्रिया की कोटि", "गति के नियम", "घर्षण").
7. Only fixed system fields stay in English: "difficulty", "question_type",
   "question_bucket" — these use their standard English values as defined below.
8. If any mixed-language content exists, keep exactly as printed.
"""
        else:
            language_block = ""

        if is_hindi:
            subtopic_lang_rule = (
                '"subtopic" must be in HINDI. Examples: "अभिक्रिया की कोटि", "गति के नियम", '
                '"घर्षण", "विद्युत धारा", "समाकलन", "प्रकाशिकी", "ऊष्मागतिकी" etc.\n'
                '  • Identify the specific concept/topic being tested and write it in Hindi.\n'
                '  • NEVER write subtopic in English when medium is Hindi.'
            )
        else:
            subtopic_lang_rule = (
                '"subtopic" must be in ENGLISH. Examples: "Order of Reaction", "Laws of Motion", '
                '"Friction", "Current Electricity", "Integration", "Optics" etc.\n'
                '  • Identify the specific concept/topic being tested and write it in English.\n'
                '  • NEVER write subtopic in Hindi when medium is English.'
            )

        subpart_rule = """
══════════════════════════════════════════════
RULE 1B — SUB-PARTS MUST STAY TOGETHER (CRITICAL — READ CAREFULLY)
══════════════════════════════════════════════
A question with sub-parts is ONE single question. NEVER split it into multiple JSON objects.

Sub-parts include ANY of these patterns inside a question:
  • Letters:  (a), (b), (c), (d)  OR  a., b., c., d.
  • Roman:    (i), (ii), (iii), (iv)  OR  i., ii., iii., iv.
  • Numbers:  (1), (2), (3), (4)  inside the question body

RULE: If a question stem is followed by sub-parts using ANY of the above patterns,
the ENTIRE question including ALL sub-parts goes into ONE JSON object.

EXAMPLES of what NOT to do:
  ✗ WRONG: Create separate JSON for "Find: (a) LCM" and separate for "(b) HCF"
  ✓ RIGHT: ONE JSON with question = "Find: (a) LCM of ... (b) HCF of ..."

  ✗ WRONG: Create separate JSON for "(i) Prove that..." and "(ii) Show that..."
  ✓ RIGHT: ONE JSON with question = "(i) Prove that... (ii) Show that..."

The Explanation field must contain answers/solutions for ALL sub-parts together.
"""

        _TH  = 'style="border:1px solid #333;padding:5px 10px;background:#dbeafe;font-weight:bold;text-align:center;vertical-align:middle"'
        _TD  = 'style="border:1px solid #333;padding:5px 10px;text-align:center;vertical-align:middle"'
        _TD_L = 'style="border:1px solid #333;padding:5px 10px;text-align:left;vertical-align:middle"'

        table_rule = f"""
══════════════════════════════════════════════
RULE 8 — TABLES & STRUCTURED LAYOUTS (CRITICAL — READ EVERY LINE)
══════════════════════════════════════════════

━━━ A. LCM / HCF DIVISION LADDER — PLAIN TEXT FORMAT ━━━
When a solution shows LCM or HCF step-by-step division, write it as PLAIN TEXT in this EXACT format:

2 | 15  20  25
  |___________
2 | 15  10  25
  |___________
3 | 15   5  25
  |___________
5 |  5   5  25
  |___________
5 |  1   1   5
  |___________
    1   1   1

STRICT RULES FOR THE LADDER:
1. Use PLAIN TEXT only. DO NOT use HTML <table>, markdown table (|---|), or any other format.
2. Each division step = one line: "divisor | num1  num2  num3"  (single space between numbers)
3. After EVERY divisor row, add a separator: "  |" followed by underscores covering ALL numbers
4. Final row (all 1s, no more divisor): indent with spaces, no pipe: "    1   1   1"
5. Align numbers in columns using spaces so columns are vertically straight.
6. Copy the ACTUAL numbers from the PDF exactly — replace the example 15,20,25 above.
7. Put the ENTIRE ladder in the "Explanation" field ONLY — NEVER in "question".
8. NEVER omit any step. Every row from the PDF must appear.

Example for LCM(12, 18):
2 | 12  18
  |_______
2 |  6   9
  |_______
3 |  3   9
  |_______
3 |  1   3
  |_______
    1   1

━━━ B. GENERAL DATA / FREQUENCY TABLES ━━━
Use HTML table (NOT plain text) for data tables, frequency tables, comparison tables:
<table class="ltable" style="border-collapse:collapse;margin:8px 0">
  <tr><th {_TH}>Header1</th><th {_TH}>Header2</th></tr>
  <tr><td {_TD}>value</td><td {_TD}>value</td></tr>
</table>
• Preserve ALL rows and ALL columns exactly from the PDF.
• Do NOT flatten data tables into comma-separated text.

━━━ C. MATCH THE COLUMN TABLES ━━━
Use HTML table:
<table class="match-col ltable" style="border-collapse:collapse;margin:8px 0">
  <tr><td style="border:1px solid #333;border-right:2px solid #555;padding:5px 10px;text-align:left">Col I item</td>
      <td {_TD_L}>Col II item</td></tr>
</table>

━━━ D. IMAGE + TABLE SIDE-BY-SIDE ━━━
If PDF shows an image next to a table:
<div class="compare-wrap"><img src="..." style="max-width:45%;height:auto"><table ...>...</table></div>

━━━ E. AXIS / GRAPH descriptions ━━━
If question refers to a graph: mention axis labels in text, e.g. "x-axis: Time (s), y-axis: Distance (m)"
"""

        prompt = f"""
You are a precise question paper digitizer. Extract EVERY question from this page into a JSON array.

{marks_hint_text}

{language_block}

{'CONTEXT: ' + context_block if context_block else ''}

══════════════════════════════════════════════
RULE 0 — FIELD SEPARATION (READ FIRST — MOST CRITICAL)
══════════════════════════════════════════════
Each JSON field has ONE specific purpose. NEVER mix content between fields:

"question"  → ONLY the question text as printed (the sentence/problem statement).
              NEVER put options, solution steps, LCM tables, or "Sol." text here.
"option1–4" → ONLY the answer choice text. NEVER put the question or explanation here.
"Answer"    → ONLY the answer number (1/2/3/4) or blank.
"Explanation" → The full solution/working: Sol. text, LCM/HCF tables, calculations, "∴ LCM = ...".

⚠️ VERY COMMON MISTAKE TO AVOID:
  ✗ WRONG: "question": "...how many soldiers? Sol. Option(4) LCM = <table>...</table>"
  ✓ RIGHT:  "question": "...how many soldiers?"
             "Explanation": "Sol. Option(4) LCM(15,20,25) = <table class='lcm-table'>...</table> = 300, so answer is 900"

══════════════════════════════════════════════
RULE 1 — COMPLETENESS + EXACT SEQUENCE (MOST IMPORTANT)
══════════════════════════════════════════════
• Copy ALL text VERBATIM — do NOT paraphrase, summarize, or skip anything.
• Extract EVERY question on this page — do not skip even one.
• Extract ALL option text completely — even if options are long paragraphs.
• Extract the FULL explanation/solution — every step, every line.

SEQUENCE RULE — CRITICAL:
The "question" field must contain ALL parts of the question in the EXACT order they appear in the PDF.
Text BEFORE a diagram + [DIAGRAM_X] + text AFTER the diagram — ALL go into "question" together.

COMMON MISTAKE — TEXT AFTER DIAGRAM GETS DROPPED:
  PDF shows:  [diagram]  then  "Refer to figure. Find the magnitude, x and y components."
  ✗ WRONG: "question": "[DIAGRAM_0]"    ← drops all text after diagram!
  ✓ RIGHT:  "question": "[DIAGRAM_0]\nRefer to figure. Find the magnitude, x and y components."

  PDF shows: "Given the triangle below" then [diagram] then "Find: (a) perimeter  (b) area"
  ✗ WRONG: "question": "Given the triangle below [DIAGRAM_0]"    ← drops sub-questions!
  ✓ RIGHT:  "question": "Given the triangle below [DIAGRAM_0]\nFind: (a) perimeter  (b) area"

RULE: After placing [DIAGRAM_X], continue reading and include ALL remaining text of that question.
NEVER stop extracting question text just because you placed a diagram marker.

{subpart_rule}

══════════════════════════════════════════════
RULE 2 — PAGE BOUNDARIES (READ THIS CAREFULLY)
══════════════════════════════════════════════
STEP 1 — BEFORE reading any numbered question, scan the very TOP of this page.
  Ask yourself: "Is there any answer, solution, or explanation text here that belongs
  to a question from the PREVIOUS page?"

  Signs that top-of-page text is a continuation from previous page:
  • Starts with "Ans.", "Answer:", "Sol.", "Solution:", "∴", "Therefore", "Hence"
  • Hindi: starts with "उत्तर:", "हल:", "अतः", "इसलिए", "∴"
  • Starts with "(A)", "(B)", "(C)", "(D)" without a question before it
  • Starts with a step in a solution — no question number before it

  IF YES → Create a fragment entry as the VERY FIRST item in your output:
    {{"questionid": "", "question": "", "Explanation": "<copy the full solution/answer text verbatim>", "Answer": "<if answer letter/number present, extract it>", ...all other fields as ""}}
  Then continue extracting the remaining numbered questions normally.

STEP 2 — If this page ENDS with an incomplete question (no answer/explanation visible):
  → Include it anyway with whatever fields are present — the system will merge it with the next page.

══════════════════════════════════════════════
RULE 3 — ALL VISUAL CONTENT (DIAGRAMS, GRAPHS, TABLES, CANVAS)
══════════════════════════════════════════════
{diagram_note}

VISUAL TYPES — every one of these must be captured with [DIAGRAM_X]:
  • Geometric shapes   — triangle, circle, rectangle, polygon, trapezium, rhombus
  • Graphs             — bar chart, line graph, pie chart, histogram, frequency polygon
  • Coordinate geometry — Cartesian axes, number line, plotted points
  • Physics diagrams   — circuit diagram, ray diagram, force diagram, pulley, spring
  • Chemistry diagrams — lab apparatus, structural formula, molecular diagram
  • Biology diagrams   — cell, organ, lifecycle diagram
  • Accounts / Finance — ledger table, balance sheet diagram
  • Drawn images       — any hand-drawn or PDF-canvas shape
  • Venn diagrams      — overlapping circles
  • Maps, flowcharts   — any non-text visual structure

══════════════════════════════════════════════
RULE 4 — LATEX (COMPLETE — EVERY MATH SYMBOL)
══════════════════════════════════════════════
DELIMITERS — STRICT:
  Inline  → \\\\( ... \\\\)       e.g.  \\\\(x^2 + y^2 = r^2\\\\)
  Display → \\\\[ ... \\\\]       e.g.  \\\\[\\\\frac{{a}}{{b}} + c\\\\]
  NEVER use $...$ or $$...$$

FRACTIONS:   \\\\(\\\\frac{{num}}{{den}}\\\\)   e.g. ½ → \\\\(\\\\frac{{1}}{{2}}\\\\)
SQRT:        \\\\(\\\\sqrt{{x}}\\\\)  nth-root: \\\\(\\\\sqrt[n]{{x}}\\\\)
             e.g. √2 → \\\\(\\\\sqrt{{2}}\\\\),  ³√8 → \\\\(\\\\sqrt[3]{{8}}\\\\)
POWERS:      \\\\(x^{{2}}\\\\)  \\\\(a^{{m+n}}\\\\)  \\\\(2^{{10}}\\\\)
SUBSCRIPT:   \\\\(x_{{1}}\\\\)  \\\\(a_{{ij}}\\\\)
GREEK:       \\\\(\\\\alpha\\\\) \\\\(\\\\beta\\\\) \\\\(\\\\gamma\\\\) \\\\(\\\\delta\\\\)
             \\\\(\\\\theta\\\\) \\\\(\\\\pi\\\\) \\\\(\\\\omega\\\\) \\\\(\\\\lambda\\\\)
             \\\\(\\\\mu\\\\) \\\\(\\\\sigma\\\\) \\\\(\\\\phi\\\\) \\\\(\\\\rho\\\\)
TRIG:        \\\\(\\\\sin\\\\theta\\\\) \\\\(\\\\cos 60^\\\\circ\\\\) \\\\(\\\\tan^{{-1}}x\\\\)
LOG:         \\\\(\\\\log_{{2}}8\\\\)  \\\\(\\\\ln x\\\\)  \\\\(\\\\log x\\\\)
VECTORS:     \\\\(\\\\vec{{a}}\\\\)  \\\\(\\\\overrightarrow{{AB}}\\\\)
INTEGRAL:    \\\\[\\\\int_{{a}}^{{b}} f(x)\\\\,dx\\\\]
SUMMATION:   \\\\[\\\\sum_{{i=1}}^{{n}} x_i\\\\]
LIMIT:       \\\\[\\\\lim_{{x \\\\to 0}} \\\\frac{{\\\\sin x}}{{x}} = 1\\\\]
MATRIX:      \\\\[\\\\begin{{pmatrix}} a & b \\\\\\\\ c & d \\\\end{{pmatrix}}\\\\]
INEQUALITY:  \\\\(\\\\leq\\\\) \\\\(\\\\geq\\\\) \\\\(\\\\neq\\\\) \\\\(\\\\approx\\\\)
ABS VALUE:   \\\\(|x|\\\\)
INFINITY:    \\\\(\\\\infty\\\\)
THEREFORE:   \\\\(\\\\therefore\\\\)   BECAUSE: \\\\(\\\\because\\\\)
PLUS-MINUS:  \\\\(\\\\pm\\\\)
DEGREE:      \\\\(90^\\\\circ\\\\)
CHEMICAL:    H₂O → \\\\(\\\\text{{H}}_{{2}}\\\\text{{O}}\\\\)
PERCENTAGE:  25\\\\% (escape the % sign)

══════════════════════════════════════════════
RULE 5 — FIELD VALUES
══════════════════════════════════════════════
Answer:
  • MCQs      → "1" / "2" / "3" / "4"  (A=1, B=2, C=3, D=4)
  • True/False → "1" for True/सत्य, "0" for False/असत्य
  • Numeric    → numeric string e.g. "42" or "3.14"
  • Others     → ""

Marks: single digit string only ("1","2","3","4","5") or "" if not mentioned.

medium: ALWAYS use "{medium_val}" for this extraction.

question_type: exactly one of:
  MCQs | True/False | Numeric | Subjective | Filling Blank |
  Assertion and Reasoning Questions ( A& R ) |
  Match the Column Question | Case Based Questions (CBQ)

FOR "Assertion and Reasoning Questions ( A& R )" — options MUST be:
  option1: "Both Assertion (A) and Reason (R) are true and Reason (R) is the correct explanation of Assertion (A)."
  option2: "Both Assertion (A) and Reason (R) are true but Reason (R) is NOT the correct explanation of Assertion (A)."
  option3: "Assertion (A) is true but Reason (R) is false."
  option4: "Assertion (A) is false but Reason (R) is true."
  (Use EXACTLY these texts — do not leave A&R options blank)

difficulty: Easy / Medium / Hard (your assessment)

question_bucket: carefully assess each question and assign exactly one of:
  Beginner | Target | Advance Climb | Must Do

══════════════════════════════════════════════
RULE 6 — SUBTOPIC (VERY IMPORTANT — DO NOT LEAVE BLANK)
══════════════════════════════════════════════
{subtopic_lang_rule}
  • NEVER leave subtopic as "" — always provide your best assessment.

══════════════════════════════════════════════
RULE 7 — PREVIOUS YEAR REFERENCE
══════════════════════════════════════════════
"previous_year" captures the source/reference tag printed near the question.
Look for patterns at END of question text or after solution:
  • - (Exercise-7.1-8) → extract: "Exercise-7.1-8"
  • - (Example-1) → extract: "Example-1"
  • JEE Main 2024, NEET 2023, (2024), [2024], NCERT, PYQ etc.
  • Extract EXACTLY as printed (without leading "- ").
  • Remove the reference tag from "question" field.
  • If not present → leave "previous_year" as "".

{table_rule}

══════════════════════════════════════════════
OUTPUT FORMAT — repeat for EVERY question:
══════════════════════════════════════════════
{{
  "questionid": "<number as printed, or empty for continuation fragment>",
  "question": "<complete verbatim question text WITH ALL SUB-PARTS (a,b,c / i,ii,iii / 1,2,3) included, with LaTeX and [DIAGRAM_X] if needed — WITHOUT any reference tag — tables as HTML>",
  "option1": "<option A full text, no prefix>",
  "option2": "<option B full text, no prefix>",
  "option3": "<option C full text, no prefix>",
  "option4": "<option D full text, no prefix>",
  "Answer": "<per rules above>",
  "Explanation": "<complete solution for ALL sub-parts — every line verbatim — tables as HTML with 1px border>",
  "course": "",
  "subjectname": "",
  "chapter": "",
  "practice": "",
  "subtopic": "<{'in Hindi' if is_hindi else 'in English'} — specific concept/topic — NEVER leave blank>",
  "medium": "{medium_val}",
  "difficulty": "Easy/Medium/Hard",
  "question_type": "<exact type string>",
  "previous_year": "<extracted reference tag or empty>",
  "marks": "<digit or empty>",
  "class": "",
  "book": "",
  "question_bucket": "<Beginner | Target | Advance Climb | Must Do>",
  "diagram_placements": []
}}

Return ONLY the JSON array. No markdown fences. No preamble. No trailing text.
CRITICAL: Ensure all text content is properly escaped for JSON. Use \\n for newlines inside strings.
Unicode/Devanagari characters must be included as-is (UTF-8), NOT as \\uXXXX escape sequences.
REMEMBER:
  1. Sub-parts (a,b,c / i,ii,iii / 1,2,3) of the SAME question = ONE JSON object. NEVER split them.
  2. LCM/HCF division ladder → PLAIN TEXT with "divisor | numbers" rows and "  |___" separators. NEVER HTML table.
  3. Data/frequency/match-column tables → HTML <table class="ltable"> with borders on every <td>/<th>.
  4. medium field = "{medium_val}" always.
  5. NEVER put ladders, solution text, or "Sol." in the "question" field — Explanation field only.
  6. SEQUENCE: [DIAGRAM_X] tags in TOP-TO-BOTTOM order. Text BEFORE diagram → before [DIAGRAM_X]. Text AFTER diagram → after [DIAGRAM_X].
  7. NEVER drop text that appears after a diagram. Include ALL question text: pre-diagram + [DIAGRAM_X] + post-diagram.
  8. If a question is ONLY a diagram with a one-line instruction below it — include BOTH: "[DIAGRAM_0]\nInstruction text here"
"""

        content_parts = [prompt, genai_types.Part.from_bytes(data=img_bytes, mime_type='image/png')]
        for pi in page_images:
            content_parts.append(genai_types.Part.from_bytes(
                data=base64.b64decode(pi['image_base64']), mime_type=pi['mime_type']
            ))

        last_raw_response = None

        for attempt_num, cfg in enumerate(_make_page_configs()):
            label = f"cfg{attempt_num + 1}"
            try:
                raw_text = _call_gemini(client, model_name, content_parts, cfg)
                if not raw_text:
                    print(f"  Page {page_num} [{label}]: empty — trying next")
                    continue
                last_raw_response = raw_text
                print(f"  Page {page_num} [{label}]: {len(raw_text)} chars received")
                parsed = clean_json_response(raw_text)
                if parsed:
                    if section_marks_hint:
                        for item in parsed:
                            if not fix_marks_field(item.get("marks", "")):
                                item["marks"] = section_marks_hint
                    if num_diagrams > 0:
                        parsed = deduplicate_diagram_placements(parsed)
                        parsed = auto_inject_missed_diagrams(parsed, page_images)
                    expanded = [
                        clean_question(
                            q, page_images=page_images,
                            user_subject=user_subject, user_course=user_course,
                            user_class=user_class, user_chapter=user_chapter,
                            user_practice=user_practice, user_book=user_book,
                            is_hindi=is_hindi
                        )
                        for q in parsed
                    ]
                    print(f"  Page {page_num}: {len(expanded)} question(s) extracted [{label}]")
                    return expanded
                print(f"  Page {page_num} [{label}]: JSON parse failed. First 300 chars:\n{raw_text[:300]}")
                time.sleep(1)
            except Exception as inner_e:
                err_str = str(inner_e)
                print(f"  Page {page_num} [{label}] ERROR: {err_str[:500]}")
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(30 * (attempt_num + 1))
                elif "500" in err_str or "503" in err_str:
                    time.sleep(3)
                continue

        if last_raw_response:
            try:
                fix_prompt = (
                    "The text below is supposed to be a valid JSON array of question objects "
                    "but has syntax errors. Fix ALL syntax errors and return ONLY the corrected "
                    "JSON array. Do NOT change any content, field names, or values — only fix syntax.\n\n"
                    "IMPORTANT: Preserve all Unicode/Devanagari characters exactly as-is.\n\n"
                    "Common issues to fix: trailing commas, missing quotes, unescaped characters, "
                    "truncated objects.\n\n"
                    f"Broken JSON:\n{last_raw_response[:12000]}\n\n"
                    "Return ONLY the valid JSON array. No markdown. No explanation."
                )
                fix_response = client.models.generate_content(
                    model=model_name, contents=[fix_prompt],
                    config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=65536),
                )
                parsed = clean_json_response(_safe_response_text(fix_response) or "")
                if parsed:
                    print(f"✅ Page {page_num}: JSON recovered via fix-prompt ({len(parsed)} questions)")
                    if section_marks_hint:
                        for item in parsed:
                            if not fix_marks_field(item.get("marks", "")):
                                item["marks"] = section_marks_hint
                    if num_diagrams > 0:
                        parsed = deduplicate_diagram_placements(parsed)
                        parsed = auto_inject_missed_diagrams(parsed, page_images)
                    expanded = []
                    for q in parsed:
                        q = clean_question(
                            q, page_images=page_images,
                            user_subject=user_subject, user_course=user_course,
                            user_class=user_class, user_chapter=user_chapter,
                            user_practice=user_practice, user_book=user_book,
                            is_hindi=is_hindi
                        )
                        expanded.append(q)
                    return expanded
            except Exception:
                pass

        print(f"⚠️ Page {page_num}: could not parse JSON after all retries")
        return []
    except Exception as e:
        print(f"Critical error on Page {page_num}:", e)
        return []


# ================================================================
# SECTION CONFIGURATOR UI
# ================================================================
def render_section_configurator(total_pages_hint=None):
    st.markdown("### 📑 Section Configuration")
    st.caption("Configure sections with page ranges and marks per question")

    col_count, _ = st.columns([1, 3])
    with col_count:
        num_sections = st.number_input("Total Sections", min_value=1, max_value=20, value=1, step=1)

    section_configs = []
    default_names = ["Section A", "Section B", "Section C", "Section D",
                     "Section E", "Section F", "Section G", "Section H"]

    st.markdown("#### Configure Each Section")
    for i in range(int(num_sections)):
        default_name = default_names[i] if i < len(default_names) else f"Section {i+1}"
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])
            with c1:
                name = st.text_input(f"Section {i+1} Name", value=default_name, key=f"sec_name_{i}")
            with c2:
                start_page = st.number_input("Start Page", min_value=1,
                    max_value=total_pages_hint or 9999, value=1, step=1, key=f"sec_start_{i}")
            with c3:
                end_page = st.number_input("End Page", min_value=1,
                    max_value=total_pages_hint or 9999,
                    value=total_pages_hint or 1, step=1, key=f"sec_end_{i}")
            with c4:
                marks = st.number_input("Marks Each", min_value=1, max_value=10,
                    value=1, step=1, key=f"sec_marks_{i}")

            if start_page > end_page:
                st.error(f"⚠️ {name}: Start page cannot be greater than end page!")

            section_configs.append({
                "name": name,
                "start_page": int(start_page),
                "end_page": int(end_page),
                "marks": str(int(marks))
            })

    if section_configs:
        st.markdown("**📋 Section Summary:**")
        for sec in section_configs:
            st.write(f"• **{sec['name']}**: pages {sec['start_page']}–{sec['end_page']} → {sec['marks']} mark(s) each")

    return section_configs


# ================================================================
# MAIN UI
# ================================================================

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "extraction_count" not in st.session_state:
    st.session_state.extraction_count = 0
if "extraction_total" not in st.session_state:
    st.session_state.extraction_total = 0
if "extraction_file" not in st.session_state:
    st.session_state.extraction_file = None
if "extraction_all_questions" not in st.session_state:
    st.session_state.extraction_all_questions = None
if "extraction_rendered_pages" not in st.session_state:
    st.session_state.extraction_rendered_pages = {}

st.markdown("""<style>
.block-container{padding-top:0.6rem!important;padding-bottom:0!important;max-width:100%!important}
header{visibility:hidden}
body.live-fs [data-testid="stSidebar"]{display:none!important}
body.live-fs header{display:none!important}
body.live-fs .block-container{padding:0.3rem 0.6rem!important;max-width:100%!important}
body.live-fs [data-testid="stVerticalBlockBorderWrapper"]{height:calc(100vh - 80px)!important}
body.live-fs iframe{height:calc(100vh - 100px)!important}
body.live-fs [data-testid="stAppViewContainer"]{overflow:hidden}
</style>""", unsafe_allow_html=True)

st.title("⚡ PDF to JSON - Complete Extraction")
st.markdown("---")

SUBJECT_OPTIONS = ["Hindi", "English", "Math", "Physics", "Chemistry", "Biology", "Social Science",
                   "हिंदी", "अंग्रेज़ी", "गणित", "भौतिक विज्ञान", "रसायन विज्ञान", "जीव विज्ञान", "सामाजिक विज्ञान"]

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    num_threads = st.slider("Threads (Concurrency)", 1, 5, 2)
    model_choice = st.selectbox("Model", [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ])
    st.info("Threads 2-3 recommended for free tier.")
    st.markdown("---")

    st.subheader("🌐 Medium / Language")
    is_hindi = st.toggle("🇮🇳 Hindi Medium PDF", value=False,
                          help="Enable this if your PDF is in Hindi. All extracted text will remain in Hindi.")
    if is_hindi:
        st.success("✅ Hindi Medium — text will be extracted in Hindi (हिंदी)")
    else:
        st.info("English Medium (default)")

    st.markdown("---")
    st.subheader("📋 Question Metadata")
    user_subject = st.selectbox("Subject Name", options=[""] + SUBJECT_OPTIONS, index=0)
    user_course = st.text_input("Course", placeholder="e.g. JEE Main, NEET, CBSE...")
    user_class = st.text_input("Class", placeholder="e.g. 10, 11, 12...")
    user_chapter = st.text_input("Chapter", placeholder="e.g. Laws of Motion, गति के नियम...")
    user_practice = st.text_input("Practice", placeholder="e.g. Unit Test-1, Exercise 2.1...")
    user_book = st.text_input("Book", placeholder="e.g. NCERT, RD Sharma, HC Verma...")
    st.markdown("---")
    st.subheader("🔖 Checkpoint Manager")
    existing = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith("_checkpoint.json")]
    if existing:
        for ck in existing:
            try:
                with open(os.path.join(CHECKPOINT_DIR, ck)) as f:
                    ck_data = json.load(f)
                done = len(ck_data.get("completed_pages", {}))
                total = ck_data.get("total_pages", "?")
                st.caption(f"📄 {ck.replace('_checkpoint.json','')} — {done}/{total} pages done")
            except Exception:
                st.caption(f"📄 {ck}")
        if st.button("🗑️ Delete All Checkpoints"):
            for ck in existing:
                os.remove(os.path.join(CHECKPOINT_DIR, ck))
            st.success("Checkpoints deleted!")
    else:
        st.caption("No checkpoints found.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.extraction_file:
    st.session_state.extraction_result = None
    st.session_state.extraction_count = 0
    st.session_state.extraction_total = 0
    st.session_state.extraction_all_questions = None
    st.session_state.extraction_rendered_pages = {}

if any([user_subject, user_course, user_class, user_chapter, user_book]):
    with st.expander("📌 Metadata Preview", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Subject", user_subject or "—")
        cols[1].metric("Course", user_course or "—")
        cols[2].metric("Class", user_class or "—")
        cols[3].metric("Chapter", user_chapter or "—")
        cols[4].metric("Book", user_book or "—")

st.markdown("---")
use_sections = st.toggle("📑 Configure PDF Sections (marks per section)", value=False)
section_configs = []
if use_sections:
    with st.container():
        st.info("Configure section page ranges and marks per question")
        section_configs = render_section_configurator(total_pages_hint=None)

st.markdown("---")

_tc1, _tc2 = st.columns([1, 4])
with _tc1:
    if st.button("🔌 Test API"):
        if not api_key:
            st.error("Enter API Key first!")
        else:
            with st.spinner("Testing API connection..."):
                try:
                    _client_test = genai.Client(api_key=api_key)
                    _cfg_test = genai_types.GenerateContentConfig(
                        temperature=0, max_output_tokens=20
                    )
                    _resp_test = _client_test.models.generate_content(
                        model=model_choice,
                        contents=["Say: OK"],
                        config=_cfg_test,
                    )
                    _txt_test = _safe_response_text(_resp_test)
                    if _txt_test:
                        st.success(f"✅ API OK — Model `{model_choice}` works! Response: `{_txt_test.strip()[:60]}`")
                    else:
                        st.warning(f"⚠️ Connected but empty response — model may not support this config")
                except Exception as _e:
                    st.error(f"❌ API FAILED: {str(_e)}")

st.markdown("---")
_tp1, _tp2 = st.columns([1, 4])
with _tp1:
    if st.button("🧪 Test Page 1", help="Extract page 1 only and show raw response"):
        if not api_key or not uploaded_file:
            st.error("Need API key and PDF first!")
        else:
            with st.spinner("Calling Gemini on page 1..."):
                try:
                    import tempfile as _tf
                    with _tf.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp:
                        _tmp.write(uploaded_file.read())
                        _tp = _tmp.name
                    uploaded_file.seek(0)

                    _tc = genai.Client(api_key=api_key)
                    _img = pdf_page_to_png_bytes(_tp, 0)
                    _cfg = genai_types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                    _r = _tc.models.generate_content(
                        model=model_choice,
                        contents=["Extract questions from this PDF page as a JSON array.",
                                  genai_types.Part.from_bytes(data=_img, mime_type="image/png")],
                        config=_cfg,
                    )
                    _txt = _safe_response_text(_r)
                    if _txt:
                        st.success(f"✅ Got {len(_txt)} chars from model")
                        with st.expander("Raw response (first 2000 chars)", expanded=True):
                            st.code(_txt[:2000])
                    else:
                        fr = "unknown"
                        try: fr = _r.candidates[0].finish_reason
                        except: pass
                        st.error(f"❌ Empty response. finish_reason={fr}")
                    import os as _os; _os.unlink(_tp)
                except Exception as _e:
                    st.error(f"❌ Error: {str(_e)}")

st.markdown("---")
if st.button("Start / Resume Extraction 🚀", type="primary"):
    if not api_key or not uploaded_file:
        st.error("API Key and PDF are required!")
        st.stop()

    if use_sections and section_configs:
        has_error = False
        for sec in section_configs:
            if sec["start_page"] > sec["end_page"]:
                st.error(f"❌ {sec['name']}: Start page > End page")
                has_error = True
        if has_error:
            st.stop()

    try:
        import traceback as _tb

        with st.status("Phase 1: Loading PDF...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            doc = fitz.open(temp_pdf_path)
            total_pages = len(doc)
            doc.close()

            skip_hashes = compute_template_image_hashes(temp_pdf_path)
            lang_label = "Hindi 🇮🇳" if is_hindi else "English"
            if skip_hashes:
                status.update(
                    label=f"Done! {total_pages} pages — {len(skip_hashes)} template image(s) filtered | Medium: {lang_label}",
                    state="complete"
                )
            else:
                status.update(
                    label=f"Done! {total_pages} pages loaded | Medium: {lang_label}",
                    state="complete"
                )

        page_marks_map = [""] * total_pages
        if use_sections and section_configs:
            page_marks_map = build_section_page_map(section_configs, total_pages)
            covered = sum(1 for m in page_marks_map if m)
            st.info(f"Section config: {covered}/{total_pages} pages have marks defined")

        page_results, already_done = load_checkpoint(uploaded_file.name, total_pages)
        if already_done > 0:
            st.info(f"Resuming from checkpoint: {already_done}/{total_pages} pages done")
        else:
            st.info("Starting fresh extraction...")

        pending_indices = [i for i, r in enumerate(page_results) if r is None]
        pending_args = [
            (temp_pdf_path, i, i + 1, api_key, model_choice,
             user_subject, user_course, user_class, user_chapter, user_practice,
             user_book, page_marks_map[i], skip_hashes, is_hindi)
            for i in pending_indices
        ]

        components.html("""
<style>
#fsb{cursor:pointer;border:1.5px solid #6366f1;border-radius:7px;
     padding:7px 16px;background:#f5f3ff;color:#4f46e5;
     font-size:13px;font-weight:600;white-space:nowrap;
     transition:all .15s;width:100%;margin-top:6px}
#fsb:hover{background:#ede9fe;border-color:#4f46e5}
</style>
<script>
function toggleFS(){
  var body=window.parent.document.body;
  var on=body.classList.toggle('live-fs');
  document.getElementById('fsb').textContent=on?'⊟ Exit Full Screen':'⛶ Full Screen';
}
</script>
<button id="fsb" onclick="toggleFS()">&#x26F6; Full Screen</button>
""", height=52)

        live_col_pdf, live_col_preview, live_col_json = st.columns([1, 1, 1])
        with live_col_pdf:
            st.markdown("**📄 Complete PDF** *(page-by-page as processed)*")
            _pdf_placeholder = st.empty()
            _pdf_info        = st.empty()
        with live_col_preview:
            st.markdown("**🔬 Preview** *(rendered LaTeX + images)*")
            _preview_placeholder = st.empty()
        with live_col_json:
            st.markdown("**📋 Complete JSON**")
            _json_placeholder = st.empty()
            _json_info        = st.empty()

        progress_bar = st.progress(already_done / total_pages if total_pages else 0)
        status_text = st.empty()
        completed = already_done

        _rendered_pages: dict = {}
        _done_pages:    set   = set()
        _page_q_counts: dict  = {}
        live_qs: list         = []

        status_text.text("Loading PDF pages…")
        for _pi in range(total_pages):
            try:
                _rendered_pages[_pi] = pdf_page_to_png_bytes(temp_pdf_path, _pi, dpi=80)
            except Exception:
                pass

        for _pi in range(total_pages):
            if page_results[_pi] is not None:
                _done_pages.add(_pi)
                _page_q_counts[_pi] = sum(
                    1 for q in page_results[_pi]
                    if isinstance(q, dict) and "_page_error" not in q
                )
                for q in (page_results[_pi] or []):
                    if isinstance(q, dict) and "_page_error" not in q:
                        live_qs.append(q)

        def _refresh_panels(current_idx=None):
            with _pdf_placeholder.container(height=900):
                for _pi in sorted(_rendered_pages.keys()):
                    if _pi in _done_pages:
                        n = _page_q_counts.get(_pi, 0)
                        cap = f"✅ Page {_pi+1} — {n} question(s)"
                    elif _pi == current_idx:
                        cap = f"⏳ Page {_pi+1} — extracting…"
                    else:
                        cap = f"Page {_pi+1}"
                    st.image(_rendered_pages[_pi], caption=cap, use_container_width=True)
            done_count = len(_done_pages)
            _pdf_info.caption(
                f"{'✅' if done_count == total_pages else '⏳'} {done_count} / {total_pages} pages done"
            )
            if live_qs:
                with _json_placeholder.container(height=900):
                    st.code(json.dumps(live_qs, indent=2, ensure_ascii=False), language="json")
                _json_info.caption(f"Questions so far: {len(live_qs)}")
            else:
                _json_placeholder.caption("*Waiting for first page…*")
            if live_qs:
                with _preview_placeholder.container(height=900):
                    components.html(_build_preview_html(live_qs), height=880, scrolling=True)

        _refresh_panels()

        if pending_args:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {
                    executor.submit(process_single_page, args): pending_indices[j]
                    for j, args in enumerate(pending_args)
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    # ── FIX 3: wrap each future result in try/except ──
                    try:
                        result = future.result()
                    except Exception as _fe:
                        print(f"⚠️ Page {idx+1} future exception: {_fe}")
                        result = []
                    page_results[idx] = result if result else []
                    _done_pages.add(idx)
                    _page_q_counts[idx] = sum(
                        1 for q in page_results[idx]
                        if isinstance(q, dict) and "_page_error" not in q
                    )
                    for q in page_results[idx]:
                        if isinstance(q, dict) and "_page_error" not in q:
                            live_qs.append(q)
                    completed += 1
                    save_checkpoint(uploaded_file.name, page_results, total_pages)
                    progress_bar.progress(completed / total_pages)
                    _total_qs = sum(_page_q_counts.values())
                    status_text.text(
                        f"✅ Page {idx+1} done — {_page_q_counts[idx]} question(s) | "
                        f"Total so far: {_total_qs}"
                    )
                    _refresh_panels(idx)
        else:
            status_text.text("All pages already processed!")

        with st.status("Phase 2: Stitching split questions...") as stitch_status:
            page_results = stitch_split_questions_enhanced(page_results)
            stitched = sum(1 for p in page_results if p)
            stitch_status.update(label=f"Stitching complete — {stitched} pages with content", state="complete")

        with st.status("Phase 3: Recovering missing answers...") as recover_status:
            page_results, recovered = recover_missing_answers(
                temp_pdf_path, page_results, api_key, model_choice,
                user_subject, user_course, user_class, user_chapter, user_practice,
                user_book, recover_status, skip_hashes=skip_hashes, is_hindi=is_hindi
            )
            if recovered:
                recover_status.update(label=f"Recovered {recovered} missing answer(s)", state="complete")
            else:
                recover_status.update(label="All answers present — nothing to recover", state="complete")

        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

        all_questions = []
        for page_idx, result in enumerate(page_results):
            if result:
                for q in result:
                    all_questions.append(q)

        if all_questions:
            all_questions = [fix_misplaced_tables(q) for q in all_questions]
            raw_count = len(all_questions)
            all_questions, duplicates_removed = final_cleanup(all_questions)
            final_count = len(all_questions)

            question_page_map = {}
            q_idx = 0
            for page_idx, result in enumerate(page_results):
                if result:
                    for _ in result:
                        question_page_map[q_idx] = page_idx
                        q_idx += 1

            all_questions = infer_marks_from_sections(
                all_questions,
                page_marks_map=page_marks_map if use_sections else None,
                question_page_map=question_page_map if use_sections else None
            )
            all_questions = fix_exam_field(all_questions)

            for idx, q in enumerate(all_questions):
                q["questionid"] = idx + 1
                q["medium"] = "Hindi" if is_hindi else "English"

            all_questions = [apply_field_order(q) for q in all_questions]

            final_json = json.dumps(all_questions, indent=4, ensure_ascii=False)

            st.session_state.extraction_result = final_json
            st.session_state.extraction_count = final_count
            st.session_state.extraction_total = total_pages
            st.session_state.extraction_file = uploaded_file.name
            st.session_state.extraction_all_questions = all_questions
            st.session_state.extraction_rendered_pages = dict(_rendered_pages)

            st.success(f"✅ Extraction Complete! **{final_count} questions** extracted")
            if is_hindi:
                st.info("🇮🇳 Hindi Medium — सभी प्रश्न हिंदी में निकाले गए हैं")
            if duplicates_removed:
                st.warning(f"🔁 {duplicates_removed} duplicate(s) auto-removed")
            else:
                st.info("✅ No duplicates found")

            incomplete = [
                q for q in all_questions
                if _question_needs_answer(q) and _s(q.get("question_type")) not in (
                    "Subjective", "Match the Column Question",
                    "Assertion and Reasoning Questions ( A& R )",
                    "Case Based Questions (CBQ)"
                )
            ]
            subj_incomplete = [
                q for q in all_questions
                if _s(q.get("question_type")) in (
                    "Subjective", "Match the Column Question",
                    "Assertion and Reasoning Questions ( A& R )",
                    "Case Based Questions (CBQ)"
                ) and not _s(q.get("Explanation"))
            ]
            all_incomplete = incomplete + subj_incomplete
            if all_incomplete:
                with st.expander(f"⚠️ {len(all_incomplete)} question(s) still missing Answer/Explanation", expanded=True):
                    for q in all_incomplete:
                        qid  = q.get("questionid", "?")
                        qtxt = _s(q.get("question"))[:120] + ("..." if len(_s(q.get("question"))) > 120 else "")
                        qtyp = q.get("question_type", "")
                        st.markdown(f"**Q{qid}** `[{qtyp}]` — _{qtxt}_")
            else:
                st.success("✅ All questions complete — nothing skipped!")

            with_marks  = sum(1 for q in all_questions if str(q.get("marks", "")).strip())
            with_py     = sum(1 for q in all_questions if str(q.get("previous_year", "")).strip())
            with_book   = sum(1 for q in all_questions if str(q.get("book", "")).strip())
            with_bucket = sum(1 for q in all_questions if str(q.get("question_bucket", "")).strip())
            with_sub    = sum(1 for q in all_questions if str(q.get("subtopic", "")).strip())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("✅ With Marks",       f"{with_marks} / {final_count}")
            c2.metric("📅 With Prev. Year",  f"{with_py} / {final_count}")
            c3.metric("📚 With Book",        f"{with_book} / {final_count}")
            c4.metric("🎯 With Bucket",      f"{with_bucket} / {final_count}")
            c5.metric("🏷️ With Subtopic",    f"{with_sub} / {final_count}")

            type_counts = {}
            for q in all_questions:
                t = q.get("question_type", "Unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            bucket_counts = {}
            for q in all_questions:
                b = q.get("question_bucket", "Unknown") or "Unknown"
                bucket_counts[b] = bucket_counts.get(b, 0) + 1

            col_left, col_right = st.columns(2)
            with col_left:
                with st.expander("📊 Question Type Breakdown"):
                    for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                        st.write(f"• **{q_type}**: {count}")

            with col_right:
                with st.expander("🎯 Question Bucket Breakdown"):
                    bucket_order = ["Beginner", "Target", "Advance Climb", "Must Do"]
                    for bucket in bucket_order:
                        count = bucket_counts.get(bucket, 0)
                        if count:
                            st.write(f"• **{bucket}**: {count}")
                    for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
                        if bucket not in bucket_order:
                            st.write(f"• **{bucket}**: {count}")

            with st.expander("Preview (First 5 Questions)"):
                st.json(all_questions[:5])

            delete_checkpoint(uploaded_file.name)
            st.caption("Checkpoint deleted successfully")

            st.markdown("---")
            if st.button("🔄 Re-extract from Scratch", type="secondary"):
                st.session_state.extraction_result = None
                delete_checkpoint(uploaded_file.name)
                st.rerun()

        else:
            st.error("No questions found. Delete checkpoint from sidebar and try again.")
            st.info("💡 Sidebar → Checkpoint Manager → Delete All Checkpoints → then re-extract.")
            with st.expander("🔍 Debug — Page 1 API test", expanded=True):
                try:
                    import tempfile as _tf2
                    with _tf2.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp2:
                        uploaded_file.seek(0)
                        _tmp2.write(uploaded_file.read())
                        _dbg_path = _tmp2.name
                    uploaded_file.seek(0)
                    _dbg_client = genai.Client(api_key=api_key)
                    _dbg_img    = pdf_page_to_png_bytes(_dbg_path, 0)
                    _dbg_cfg    = genai_types.GenerateContentConfig(
                        temperature=0, max_output_tokens=512)
                    _dbg_resp   = _dbg_client.models.generate_content(
                        model=model_choice,
                        contents=["Describe what you see in this image in 2 sentences.",
                                  genai_types.Part.from_bytes(data=_dbg_img, mime_type="image/png")],
                        config=_dbg_cfg,
                    )
                    _dbg_txt = _safe_response_text(_dbg_resp)
                    import os as _os2; _os2.unlink(_dbg_path)
                    if _dbg_txt:
                        st.success(f"✅ Model sees the page. Response:\n\n{_dbg_txt}")
                    else:
                        st.error("Model returned empty — API or model issue.")
                except Exception as _dbg_e:
                    st.error(f"Debug failed: {_dbg_e}")

            if st.button("🔄 Retry Extraction", type="primary"):
                st.session_state.extraction_result = None
                delete_checkpoint(uploaded_file.name)
                st.rerun()

    except Exception as global_error:
        # ── FIX 4: show full traceback so exact line is visible in logs ──
        import traceback as _tb
        _tb_str = _tb.format_exc()
        print(_tb_str)
        st.error(f"Fatal Error: {str(global_error)}")
        st.code(_tb_str, language="python")
        st.warning("Progress has been saved. Click 'Start / Resume' to continue.")


# ================================================================
# PERSISTENT RESULT PANELS
# ================================================================
_pr_qs    = st.session_state.get("extraction_all_questions")
_pr_json  = st.session_state.get("extraction_result")
_pr_pages = st.session_state.get("extraction_rendered_pages", {})
_pr_same  = (st.session_state.extraction_file == (uploaded_file.name if uploaded_file else None))

if _pr_qs and _pr_json and _pr_same:
    st.markdown("---")
    _pr_col_pdf, _pr_col_prev, _pr_col_json = st.columns([1, 1, 1])
    with _pr_col_pdf:
        st.markdown("**📄 PDF Pages**")
        with st.container(height=900):
            if _pr_pages:
                for _pi in sorted(_pr_pages.keys()):
                    st.image(_pr_pages[_pi], caption=f"Page {_pi+1}", use_container_width=True)
            else:
                st.caption("PDF thumbnails not available")
    with _pr_col_prev:
        st.markdown("**🔬 Preview**")
        with st.container(height=900):
            components.html(_build_preview_html(_pr_qs), height=880, scrolling=True)
    with _pr_col_json:
        st.markdown("**📋 Final JSON**")
        with st.container(height=900):
            st.code(_pr_json, language="json")

# ================================================================
# PERSISTENT DOWNLOAD SECTION
# ================================================================
st.markdown("---")
all_pages_done = (
    st.session_state.extraction_result is not None and
    st.session_state.extraction_total > 0 and
    st.session_state.extraction_file == (uploaded_file.name if uploaded_file else None)
)

if all_pages_done:
    st.success(
        f"📦 Ready to download — **{st.session_state.extraction_count} questions** "
        f"from **{st.session_state.extraction_total} pages** ({st.session_state.extraction_file})"
    )
    st.download_button(
        "📥 Download JSON",
        data=st.session_state.extraction_result.encode("utf-8"),
        file_name="questions.json",
        mime="application/json; charset=utf-8",
        type="primary",
        use_container_width=True,
    )
else:
    st.download_button(
        "📥 Download JSON  (process all pages first)",
        data="",
        file_name="questions.json",
        mime="application/json",
        disabled=True,
        use_container_width=True,
    )
