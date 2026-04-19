import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
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

# ---------------- CONFIG ----------------
CHECKPOINT_DIR = "checkpoints"

st.set_page_config(
    page_title="Ultra-Fast Extraction: PDF to JSON",
    page_icon="⚡",
    layout="wide"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# canonical field order
FIELD_ORDER = [
    "questionid", "question", "option1", "option2", "option3", "option4",
    "Answer", "Explanation", "course", "subjectname", "chapter", "practice",
    "subtopic", "medium", "difficulty", "question_type", "previous_year",
    "marks", "class",
]

# ================================================================
# DISPLAY HELPER
# ================================================================
def format_text_with_line_breaks(text):
    if not text or not isinstance(text, str):
        return text
    return text.replace('\n', '<br>')

def display_question_with_breaks(q):
    result = []
    if q.get('question'):
        result.append(f"**Question:** {format_text_with_line_breaks(q['question'])}")
    for i in range(1, 5):
        opt = q.get(f'option{i}')
        if opt:
            result.append(f"**Option {i}:** {format_text_with_line_breaks(opt)}")
    if q.get('Answer'):
        result.append(f"**Answer:** {q['Answer']}")
    if q.get('Explanation'):
        result.append(f"**Explanation:** {format_text_with_line_breaks(q['Explanation'])}")
    return '<br><br>'.join(result)

# ================================================================
# PDF HELPERS (PyMuPDF)
# ================================================================
def pdf_page_to_png_bytes(pdf_path: str, page_num: int, dpi: int = 200) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes

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

def extract_page_embedded_images(pdf_path: str, page_num: int,
                                 skip_hashes: set | None = None) -> list[dict]:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            img_data = doc.extract_image(xref)
            raw = img_data["image"]
            if skip_hashes:
                img_hash = hashlib.md5(raw).hexdigest()
                if img_hash in skip_hashes:
                    continue
            w = img_data.get("width", 0)
            h = img_data.get("height", 0)
            if w < 80 or h < 50:
                continue
            ext = img_data.get("ext", "png")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            b64 = base64.b64encode(raw).decode()
            images.append({
                "mime_type": mime,
                "image_base64": b64,
                "data_uri": f"data:{mime};base64,{b64}",
            })
        except Exception:
            pass
    doc.close()
    return images

def _img_tag(data_uri: str, width: int = 280) -> str:
    # responsive image: max-width 100% of container, plus a max width in pixels
    return f'<img src="{data_uri}" style="max-width:100%; max-width:{width}px; display:block; margin:6px 0; height:auto;" loading="lazy"/>'

def inject_diagrams_into_question(q: dict, page_images: list[dict]) -> dict:
    """Inject diagrams at the correct position using [DIAGRAM_X] markers or heuristics."""
    if not page_images:
        q.pop("diagram_placements", None)
        q.pop("diagram_image_indices", None)
        return q

    # First pass: explicit markers [DIAGRAM_X]
    diagram_injected = False
    for field in ["question", "Explanation", "option1", "option2", "option3", "option4"]:
        if field in q and q[field]:
            matches = re.findall(r'\[DIAGRAM_(\d+)\]', q[field])
            for match in matches:
                idx = int(match)
                if 0 <= idx < len(page_images):
                    tag = _img_tag(page_images[idx]["data_uri"], width=300)
                    q[field] = q[field].replace(f'[DIAGRAM_{idx}]', tag)
                    diagram_injected = True
                else:
                    q[field] = q[field].replace(f'[DIAGRAM_{idx}]', '')
            # Also handle generic [DIAGRAM]
            if '[DIAGRAM]' in q[field] and not diagram_injected:
                # find first unused diagram
                used = set()
                for f in ["question", "Explanation", "option1", "option2", "option3", "option4"]:
                    for m in re.findall(r'\[DIAGRAM_(\d+)\]', q.get(f, '')):
                        used.add(int(m))
                for idx in range(len(page_images)):
                    if idx not in used:
                        tag = _img_tag(page_images[idx]["data_uri"], width=300)
                        q[field] = q[field].replace('[DIAGRAM]', tag)
                        diagram_injected = True
                        break

    placements = q.pop("diagram_placements", [])
    indices_fallback = q.pop("diagram_image_indices", [])

    if placements and not diagram_injected:
        # sort placements by position priority: start, end, replace
        placements_sorted = sorted(placements, key=lambda p: (p.get("position", "end") == "start", p.get("position", "end") == "replace", p.get("position", "end") == "end"), reverse=True)
        for p in placements_sorted:
            idx = p.get("image_index", 0)
            field = p.get("field", "Explanation")
            position = p.get("position", "end")
            if 0 <= idx < len(page_images) and field in q:
                tag = _img_tag(page_images[idx]["data_uri"], width=300)
                if position == "start":
                    q[field] = tag + "<br>" + (q.get(field) or "")
                elif position == "replace" and "[DIAGRAM]" in q.get(field, ""):
                    q[field] = q[field].replace("[DIAGRAM]", tag)
                else:
                    q[field] = (q.get(field) or "") + "<br>" + tag
                diagram_injected = True
                break

    # Fallback: heuristic injection
    if not diagram_injected and indices_fallback:
        question_text = q.get("question", "").lower()
        explanation_text = q.get("Explanation", "").lower()
        diagram_keywords = ['figure', 'fig.', 'diagram', 'image', 'shown below',
                            'given below', 'above figure', 'graph', 'circuit',
                            'structure', 'photo', 'picture', 'illustration']
        for idx in indices_fallback:
            if 0 <= idx < len(page_images):
                if any(keyword in question_text for keyword in diagram_keywords):
                    tag = _img_tag(page_images[idx]["data_uri"], width=300)
                    q["question"] = (q.get("question") or "") + f"<br>{tag}"
                    diagram_injected = True
                    break
                elif any(keyword in explanation_text for keyword in diagram_keywords):
                    tag = _img_tag(page_images[idx]["data_uri"], width=300)
                    q["Explanation"] = (q.get("Explanation") or "") + f"<br>{tag}"
                    diagram_injected = True
                    break

    # Clean up leftover markers
    for field in ["question", "Explanation", "option1", "option2", "option3", "option4"]:
        if field in q and q[field]:
            q[field] = re.sub(r'\[DIAGRAM_\d+\]', '', q[field])
            q[field] = re.sub(r'\[DIAGRAM\]', '', q[field])

    return q

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
            str(i): data for i, data in enumerate(page_results) if data is not None
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
    text = re.sub(r'<img\s+[^>]*src=["\']data:[^"\']*["\'][^>]*/?>', '[IMAGE]',
                  text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '', text)
    return text.strip()

def clean_explanation_prefix(text):
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    prefix_pattern = re.compile(
        r'^(?:Ans(?:wer)?\.?\s*:?\s*(?:\([A-Da-d]\))?\s*|Sol(?:ution)?\.?\s*:?\s*)',
        re.IGNORECASE
    )
    for _ in range(3):
        cleaned = prefix_pattern.sub('', text).strip()
        if cleaned == text:
            break
        if not cleaned:
            break
        text = cleaned
    return text

def scrub_references_from_question(text):
    """Remove inline source references like '- (Exerise-2.1-1)' from question text."""
    if not text or not isinstance(text, str):
        return text
    pattern = re.compile(
        r'(?:<br\s*/?>|\n)*\s*[-\u2013\u2014]*\s*\(\s*(?:Exercise|Exerise|Ex|NCERT|Example|Year|Quest|Page)[\s\S]*?\)',
        re.IGNORECASE
    )
    text = pattern.sub('', text)
    text = re.compile(r'\s*[\(\[]\s*(?:19|20)\d{2}\s*[\)\]]$').sub('', text)
    text = re.sub(r'(\\+)\)\)+$', r'\1', text)
    text = re.sub(r'(\.\s*)\)+$', r'\1', text)
    return text.strip()

# ================================================================
# PREVIOUS_YEAR EXTRACTION
# ================================================================
_PREV_YEAR_PATTERNS = [
    re.compile(
        r'\(\s*((?:Misc(?:ellaneous)?\s+)?(?:Exercise|Exerise|Ex|Example|NCERT)\s*[-–]?\s*[\d\.]+(?:\s*\([\d]+\))?)\s*\)',
        re.IGNORECASE
    ),
    re.compile(
        r'\b((?:Misc(?:ellaneous)?\s+)?(?:Exercise|Exerise|Ex|Example|NCERT)\s*[-–]?\s*[\d\.]+(?:\s*\([\d]+\))?)\b',
        re.IGNORECASE
    ),
    re.compile(r'[\(\[]\s*((?:19|20)\d{2})\s*[\)\]]'),
]

def extract_previous_year_from_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text_stripped = text.strip()
    for pat in _PREV_YEAR_PATTERNS:
        m = pat.search(text_stripped)
        if m:
            val = m.group(1).strip()
            val = re.sub(r'\s*[-–]\s*', '-', val)
            val = re.sub(r'\s+', ' ', val)
            return val
    return ""

def fix_previous_year_field(py_val):
    if not py_val or not isinstance(py_val, str):
        return ""
    val = py_val.strip()
    garbage = {
        'null', 'none', 'n/a', 'not applicable', 'not found',
        'not available', 'unknown', 'na', '-', '', 'nil',
        'none', 'empty', 'blank', 'undefined', 'nan'
    }
    if val.lower() in garbage:
        return ""
    val = re.sub(r'!\[.*?\]\(.*?\)', '', val)
    val = re.sub(r'\[.*?\]\(.*?\)', '', val)
    val = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '', val)
    val = val.strip()
    if not val:
        return ""
    if len(val) == 1 and not val.isalnum():
        return ""
    if re.search(r'[A-Za-z0-9]', val):
        return val
    return ""

def enrich_previous_year(q: dict) -> dict:
    py = fix_previous_year_field(q.get("previous_year", ""))
    if not py:
        raw_q = q.get("question", "") or ""
        py = extract_previous_year_from_text(raw_q)
    if not py:
        raw_e = q.get("Explanation", "") or ""
        py = extract_previous_year_from_text(raw_e)
    if py:
        q["previous_year"] = py
    return q

# ================================================================
# BOUNDARY DETECTION AND MERGING
# ================================================================
def text_continues_naturally(prev_text, next_text):
    if not prev_text or not next_text:
        return False
    prev_end = prev_text.strip()[-50:] if len(prev_text) > 50 else prev_text.strip()
    next_start = next_text.strip()[:50] if len(next_text) > 50 else next_text.strip()
    continuation_words = [
        'as', 'the', 'and', 'or', 'but', 'so', 'therefore', 'hence', 'thus',
        'then', 'also', 'if', 'when', 'where', 'which', 'that', 'this', 'these',
        'those', 'its', 'their', 'from', 'with', 'without', 'by', 'for', 'of',
        'to', 'in', 'on', 'at', 'since', 'because', 'while', 'although'
    ]
    first_word = next_start.split()[0].lower() if next_start.split() else ""
    ends_mid = prev_end[-1] not in '.!?'
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
    r'(?:Ans(?:wer)?|Sol(?:ution)?)[.\s:→]*'
    r'(?:\(?[A-D1-4]\)?)?\s*'
    r'(?:.*?\n)+?'
    r'\s*(\d+[\.\)]\s+[A-Z\(]|Q\.?\s*\d+[\.\)])',
    re.IGNORECASE | re.DOTALL
)

_ANS_BLOCK_RE = re.compile(
    r'\n?\s*(?:Ans(?:wer)?|Sol(?:ution)?)[.\s:→]*(\(?[A-D1-4]\)?)?\s*',
    re.IGNORECASE
)

_Q_NUMBER_RE = re.compile(
    r'(?:^|\n)\s*(\d+[\.\)]\s+|\([ivxIVX\d]+\)\s+|Q\.?\s*\d+[\.\)]\s*)',
    re.MULTILINE
)

def split_combined_questions(q):
    question_text = _s(q.get("question"))
    if not question_text:
        return [q]
    if not _NEW_Q_AFTER_ANS_RE.search(question_text):
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
        new_q["question"] = seg
        new_q["Answer"] = ""
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
                r'^(?:Sol(?:ution)?|Explanation)[.\s:→]*',
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
    r'correct\s+(?:option|answer)|the\s+answer)[.\s:→]*',
    re.IGNORECASE
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
        r'^(?:ans(?:wer)?[.\s:→]*)\(?([A-D1-4])\)?\s*[,.]?\s*',
        qtext, re.IGNORECASE
    )
    if ans_match:
        frag["Answer"] = frag.get("Answer") or ans_match.group(1)
        rest = qtext[ans_match.end():].strip()
        rest = re.sub(r'^(?:sol(?:ution)?|explanation)[.\s:→]*', '', rest, flags=re.IGNORECASE).strip()
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
        merged["question"] = (base_question + "<br>" + cont_question).strip() if base_question else cont_question
    base_exp = _s(merged.get("Explanation"))
    cont_exp = _s(continuation_q.get("Explanation"))
    if cont_exp:
        merged["Explanation"] = (base_exp + "<br>" + cont_exp).strip() if base_exp else cont_exp
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
    return merged

# ================================================================
# MARKS / ANSWER / FIELD FIXES
# ================================================================
_SECTION_EACH_RE = re.compile(
    r'(\d+)\s*[Mm]arks?\s+[Ee]ach|'
    r'[Ee]ach\s+(?:of\s+)?(\d+)\s*[Mm]arks?|'
    r'questions?\s+of\s+(\d+)\s*[Mm]arks?',
)

def extract_section_marks_from_text(text):
    if not text:
        return ""
    m = _SECTION_EACH_RE.search(str(text))
    if m:
        return m.group(1) or m.group(2) or m.group(3) or ""
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
    if "true" in q_type or "false" in q_type:
        if not answer_val or not isinstance(answer_val, str):
            return ""
        val = answer_val.strip().lower()
        if val in ("true", "1", "t", "yes"):
            return "1"
        if val in ("false", "0", "f", "no"):
            return "0"
        return answer_val
    if "numeric" in q_type or "integer" in q_type:
        if not answer_val and answer_val != 0:
            return ""
        val = str(answer_val).strip()
        try:
            return str(int(float(val)))
        except Exception:
            m = re.search(r'\d+', val)
            return m.group(0) if m else val
    if "mcq" in q_type or "multiple" in q_type:
        if not answer_val or not isinstance(answer_val, str):
            return ""
        val = answer_val.strip()
        if val in ('1', '2', '3', '4'):
            return val
        letter_map = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}
        inner = re.sub(r'[()]', '', val).strip().upper()
        if inner in letter_map:
            return letter_map[inner]
        if inner in ('1', '2', '3', '4'):
            return inner
        return answer_val
    return str(answer_val).strip() if answer_val else ""

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

# ================================================================
# QUESTION TYPE NORMALISER
# ================================================================
def normalize_question_type(q_type_raw):
    q_type = str(q_type_raw).strip()
    lower  = q_type.lower()
    if "true" in lower or "false" in lower:
        return "True/False"
    elif "assertion" in lower:
        return "Assertion and Reasoning Questions ( A & R )"
    elif "match" in lower:
        return "Match the Column Question"
    elif "case" in lower:
        return "Case Based Questions (CBQ)"
    elif "blank" in lower or "filling" in lower or "fill" in lower:
        return "Filling Blank"
    elif "numeric" in lower or "integer" in lower:
        return "Numeric"
    elif "subjective" in lower:
        return "Subjective"
    else:
        return "MCQs"

def normalize_difficulty(diff_raw):
    diff = str(diff_raw or "").strip().lower()
    if not diff or diff in ("null", "none", ""):
        return "Easy"
    
    # Map to only Easy, Medium, Hard
    if diff in ("beginner", "easy", "simple", "basic", "elementary", "fundamental"):
        return "Easy"
    elif diff in ("medium", "moderate", "intermediate", "target", "average", "standard"):
        return "Medium"
    elif diff in ("hard", "difficult", "advanced", "advance", "tough", "challenging", 
                  "complex", "advance climb", "must do questions", "must do", 
                  "must-do", "must do question", "mustdo", "important", "high priority",
                  "exam favourite", "exam favorite", "frequently asked"):
        return "Hard"
    else:
        # Try to detect difficulty from text
        if any(word in diff for word in ["easy", "simple", "basic"]):
            return "Easy"
        elif any(word in diff for word in ["medium", "moderate", "intermediate"]):
            return "Medium"
        elif any(word in diff for word in ["hard", "difficult", "tough", "advanced", "challenging"]):
            return "Hard"
        else:
            return "Easy"

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
        if ans in ("true", "1", "t", "yes") or (ans and "true" in ans and "false" not in ans):
            q["Answer"] = "1"
        elif ans in ("false", "0", "f", "no") or (ans and "false" in ans):
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
        "Assertion and Reasoning Questions ( A & R )",
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

# ================================================================
# UNIFIED CLEAN_QUESTION
# ================================================================
def clean_question(q, page_images=None,
                   user_subject="", user_course="", user_class="",
                   user_chapter="", user_practice=""):
    math_fields     = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    metadata_fields = ["subjectname", "chapter", "practice", "subtopic", "medium",
                       "difficulty", "question_type", "course"]

    for k in list(q.keys()):
        if q[k] is None:
            q[k] = ""

    q.pop("category", None)
    q["question_type"] = normalize_question_type(q.get("question_type") or "MCQs")
    q["difficulty"]    = normalize_difficulty(q.get("difficulty") or "")

    if "question" in q:
        q["question"] = scrub_references_from_question(q["question"])
        q["question"] = remove_base64_images(q["question"])

    for field in list(q.keys()):
        if isinstance(q[field], str):
            q[field] = remove_cite_tags(q[field])

    if "Explanation" in q:
        q["Explanation"] = clean_explanation_prefix(q["Explanation"])

    for field in math_fields:
        if field in q:
            q[field] = convert_dollar_to_latex(q[field])

    if "question" in q:
        q["question"] = remove_base64_images(q["question"])

    if "Answer" in q:
        options = {
            "A": q.get("option1") or "", "B": q.get("option2") or "",
            "C": q.get("option3") or "", "D": q.get("option4") or "",
        }
        q["Answer"] = fix_answer_field(q["Answer"], options, q["question_type"])

    q["marks"] = fix_marks_field(q.get("marks") or "")

    q = enrich_previous_year(q)
    q["previous_year"] = fix_previous_year_field(q.get("previous_year") or "")

    for field in metadata_fields:
        if field in q and is_garbage_field(q[field], field):
            q[field] = ""

    q = enforce_question_type_rules(q)

    if page_images:
        q = inject_diagrams_into_question(q, page_images)
    else:
        q.pop("diagram_placements", None)
        q.pop("diagram_image_indices", None)

    if user_subject:  q["subjectname"] = user_subject
    if user_course:   q["course"]      = user_course
    if user_class:    q["class"]       = user_class
    if user_chapter:  q["chapter"]     = user_chapter
    q["practice"] = user_practice if user_practice else q.get("practice", "")

    newline_fields = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    for field in newline_fields:
        if field in q and isinstance(q[field], str):
            q[field] = q[field].replace('\n', '<br>')

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
                print(f"⚠️ JSON repaired via structure-close: {len(result) if isinstance(result, list) else 1} item(s)")
            return result
        except Exception:
            return None

    start_idx = text.find('[')
    end_idx   = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        result = _try_parse(text[start_idx:end_idx + 1])
        if result and isinstance(result, list):
            return result
        result = _try_parse(text[start_idx:])
        if result and isinstance(result, list):
            return result

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
        print(f"⚠️ JSON recovered {len(objects)} object(s) via individual extraction")
        return objects

    print("⚠️ JSON Parse failed — could not recover")
    return None

# ================================================================
# STITCHING
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
            if not page_results[page_idx]:
                continue
            next_idx = page_idx + 1
            while next_idx < num_pages and not page_results[next_idx]:
                next_idx += 1
            if next_idx >= num_pages:
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
                                  user_chapter, user_practice, skip_hashes=None):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        img_bytes = pdf_page_to_png_bytes(pdf_path, page_index, dpi=200)
        page_images = extract_page_embedded_images(pdf_path, page_index, skip_hashes=skip_hashes)

        prev_q_text = str(prev_last_q.get("question", ""))[:500]
        prev_q_expl = str(prev_last_q.get("Explanation", ""))[:300]

        prompt = f""" You are recovering a MISSING answer/solution for an incomplete question.

INCOMPLETE QUESTION (from previous page): "{prev_q_text}"
{f'Partial solution so far: "{prev_q_expl}"' if prev_q_expl else '(No solution extracted yet)'}

YOUR TASK: Scan this page image carefully. Find the answer and/or solution that belongs to the above question.

STRICT RULES:
1. Copy text EXACTLY as it appears — word for word.
2. Do NOT rephrase, summarize, shorten, or add anything.
3. The answer/solution will appear BEFORE the first new numbered question on this page.
4. Extract EVERY line of the solution — do not skip any steps.
5. Convert all math to LaTeX: inline \\\\( ... \\\\), display \\\\[ ... \\\\]
6. If answer letter found (A/B/C/D), map: A→1, B→2, C→3, D→4
7. IMPORTANT: If you see a reference like "- (Exerise-2.1-1)" or "- (Example-7)" near this content,
   extract it into "previous_year" field exactly as written (without the leading "- ").

Return ONLY this JSON:
{{
  "Answer": "<1/2/3/4 or 1/0 for True/False or numeric, or empty if not found>",
  "Explanation": "<exact verbatim solution text>",
  "question": "<verbatim continuation of question text if split, else empty>",
  "option1": "<verbatim continuation of option A if split, else empty>",
  "option2": "<verbatim continuation of option B if split, else empty>",
  "option3": "<verbatim continuation of option C if split, else empty>",
  "option4": "<verbatim continuation of option D if split, else empty>",
  "previous_year": "<exercise/example reference if found, else empty>"
}}

If nothing related to this question appears on this page, return: {{}} """

        for attempt in range(3):
            try:
                response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': img_bytes}])
                text = response.text.strip()
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
                        "previous_year": result.get("previous_year", ""),
                        "course": "", "subjectname": "", "chapter": "", "practice": "",
                        "subtopic": "", "medium": "English", "difficulty": "",
                        "question_type": "", "marks": "", "class": "",
                        "diagram_placements": []
                    }
                    fragment = clean_question(
                        fragment, page_images=page_images,
                        user_subject=user_subject, user_course=user_course,
                        user_class=user_class, user_chapter=user_chapter,
                        user_practice=user_practice
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
                            user_chapter, user_practice, status_placeholder,
                            skip_hashes=None):
    num_pages = len(page_results)
    recovered = 0

    targets = []
    for page_idx in range(num_pages):
        page = page_results[page_idx]
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
                            user_chapter, user_practice, skip_hashes=skip_hashes
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
# AUTO-INJECT MISSED DIAGRAMS
# ================================================================
def auto_inject_missed_diagrams(questions, page_images):
    if not questions or not page_images:
        return questions

    placed = set()
    for q in questions:
        for field in ["question", "Explanation", "option1", "option2", "option3", "option4"]:
            text = q.get(field, "") or ""
            for m in re.finditer(r'\[DIAGRAM_(\d+)\]', text):
                placed.add(int(m.group(1)))

    missing = [i for i in range(len(page_images)) if i not in placed]
    if not missing:
        return questions

    diagram_kws = ['figure', 'fig.', 'fig ', 'diagram', 'image', 'shown', 'given',
                   'above', 'below', 'graph', 'circuit', 'structure']
    for idx in missing:
        tag = f'[DIAGRAM_{idx}]'
        injected = False
        for q in questions:
            q_text = (q.get("question", "") or "").lower()
            if any(kw in q_text for kw in diagram_kws) and tag not in (q.get("question", "") or ""):
                q["question"] = (q.get("question", "") or "") + f"<br>{tag}"
                injected = True
                break
        if not injected:
            if questions:
                last = questions[-1]
                last["Explanation"] = (last.get("Explanation", "") or "") + f"<br>{tag}"

    return questions

# ================================================================
# ENHANCED GEMINI PAGE PROCESSOR
# ================================================================
def process_single_page(args):
    (pdf_path, page_index, page_num, api_key, model_name,
     user_subject, user_course, user_class, user_chapter, user_practice,
     section_marks_hint, skip_hashes) = args
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )

        img_bytes = pdf_page_to_png_bytes(pdf_path, page_index, dpi=250)
        page_images = extract_page_embedded_images(pdf_path, page_index, skip_hashes=skip_hashes)
        num_diagrams = len(page_images)

        if num_diagrams > 0:
            diagram_note = (
                f'- This page has {num_diagrams} diagram/image(s) embedded, indexed 0 to {num_diagrams - 1}.\n'
                f'  MANDATORY: Place [DIAGRAM_X] (X = index) EXACTLY at the position in the text where the diagram appears.\n'
                f'  NEVER omit a diagram. Every diagram index 0–{num_diagrams - 1} must appear exactly once.'
            )
        else:
            diagram_note = '- No embedded diagrams on this page.'

        marks_hint_text = ""
        if section_marks_hint:
            marks_hint_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION MARKS: ALL questions on this page carry {section_marks_hint} mark(s).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        prompt = f""" You are a precise question paper digitizer. Extract EVERY question from this page into a JSON array.

{marks_hint_text}

══════════════════════════════════════════════
RULE 1 — COMPLETENESS (MOST IMPORTANT)
══════════════════════════════════════════════
• Copy ALL text VERBATIM — do NOT paraphrase, summarize, or abbreviate.
• Extract EVERY question on this page — do not skip even one.
• Extract ALL option text completely — even if options are long paragraphs.
• Extract the FULL explanation/solution — every step, every line.
• If a question has sub-parts (i), (ii), (iii) — include ALL sub-parts in "question".

══════════════════════════════════════════════
RULE 2 — PAGE BOUNDARIES
══════════════════════════════════════════════
STEP 1 — Before reading any numbered question, scan the very TOP of this page.
  If there is answer/solution/explanation text that belongs to a question from the PREVIOUS page:
  → Create a fragment entry as the VERY FIRST item:
    {{"questionid": "", "question": "", "Explanation": "<full solution verbatim>", "Answer": "<letter→number>", ...all other fields empty}}
  Then continue extracting remaining numbered questions normally.

STEP 2 — If this page ENDS with an incomplete question (no answer visible):
  → Include it anyway — the system will merge with the next page.

══════════════════════════════════════════════
RULE 3 — PREVIOUS YEAR / SOURCE REFERENCE ⭐ IMPORTANT ⭐
══════════════════════════════════════════════
Many questions on this page have a source reference tag below or near them. These look like:
  - (Exerise-2.1-1)
  - (Exercise-2.2-3)
  - (Example-7(1))
  - (Miscellaneous Exercise-9(2))
  - (2019)
  - [JEE 2020]

EXTRACTION RULES FOR previous_year:
1. ALWAYS look for such a reference tag near each question (above, below, or inline).
2. Extract the reference EXACTLY as it appears (e.g. "Exerise-2.1-1", "Example-7(1)").
3. Strip any leading "- " or surrounding brackets: extract only the inner text.
4. Put this value into the "previous_year" field of that question.
5. Do NOT put it anywhere else (not in question text, not in Explanation).
6. If no reference is found for a question, set "previous_year": "".

══════════════════════════════════════════════
RULE 4 — DIAGRAMS
══════════════════════════════════════════════
{diagram_note}

══════════════════════════════════════════════
RULE 5 — LATEX
══════════════════════════════════════════════
• Inline math  → \\\\( ... \\\\)
• Display math → \\\\[ ... \\\\]
• NEVER use $...$ or $$...$$
• Convert ALL mathematical expressions to LaTeX.

══════════════════════════════════════════════
RULE 6 — FIELD VALUES
══════════════════════════════════════════════
Answer:
  • MCQs    → "1" / "2" / "3" / "4"  (A=1, B=2, C=3, D=4)
  • True/False → "1" for True, "0" for False
  • Numeric   → numeric string e.g. "42"
  • Others    → ""

Marks: single digit string only ("1","2","3","4","5") or "" if not mentioned.

question_type: exactly one of:
  MCQs | True/False | Numeric | Subjective | Filling Blank |
  Assertion and Reasoning Questions ( A & R ) |
  Match the Column Question | Case Based Questions (CBQ)

difficulty: exactly one of these three values only:
  "Easy" | "Medium" | "Hard"
  
  • Easy     → very basic, direct formula/definition recall, first-time learner level
  • Medium   → moderate difficulty, requires 1-2 steps of thinking or concept application
  • Hard     → complex, multi-step, tricky or high-level concept application

subtopic: ⭐ MANDATORY — based on the actual question content, identify the specific sub-topic.
  Examples for Maths Ch2 Relations & Functions:
    "Ordered Pairs", "Cartesian Product of Sets", "Relations", "Domain and Range",
    "Functions", "Types of Functions", "Algebra of Functions", "Graph of Functions"
  Examples for Physics:
    "Newton's First Law", "Projectile Motion", "Ohm's Law", "Magnetic Field"
  Examples for Chemistry:
    "Mole Concept", "Periodic Table", "Chemical Bonding", "Equilibrium"
  Rules:
  - Read the question carefully and assign the most specific sub-topic that fits.
  - Do NOT leave subtopic empty — always fill it based on what the question is testing.
  - Keep it short (2–5 words), title-case.

══════════════════════════════════════════════
OUTPUT FORMAT — repeat for EVERY question:
══════════════════════════════════════════════
{{
  "questionid": "<number as printed, or empty for continuation fragment>",
  "question": "<complete verbatim question text with LaTeX>",
  "option1": "<option A full text>",
  "option2": "<option B full text>",
  "option3": "<option C full text>",
  "option4": "<option D full text>",
  "Answer": "<per rules above>",
  "Explanation": "<complete solution — every line verbatim>",
  "course": "",
  "subjectname": "",
  "chapter": "",
  "practice": "",
  "subtopic": "<specific sub-topic based on question content — NEVER empty>",
  "medium": "English",
  "difficulty": "Easy/Medium/Hard",
  "question_type": "<exact type string>",
  "previous_year": "<exercise/example/year reference exactly as in image, or empty>",
  "marks": "<digit or empty>",
  "class": "",
  "diagram_placements": []
}}

Return ONLY the JSON array. No markdown fences. No preamble. No trailing text.
"""

        content_parts = [prompt, {'mime_type': 'image/png', 'data': img_bytes}]
        for pi in page_images:
            content_parts.append({
                'mime_type': pi['mime_type'],
                'data': base64.b64decode(pi['image_base64'])
            })

        last_raw_response = None

        for attempt in range(3):
            try:
                response = model.generate_content(content_parts)
                last_raw_response = response.text
                parsed = clean_json_response(response.text)
                if parsed:
                    if section_marks_hint:
                        for item in parsed:
                            existing = fix_marks_field(item.get("marks", ""))
                            if not existing:
                                item["marks"] = section_marks_hint

                    if num_diagrams > 0:
                        parsed = auto_inject_missed_diagrams(parsed, page_images)

                    expanded = []
                    for q in parsed:
                        q = clean_question(
                            q, page_images=page_images,
                            user_subject=user_subject, user_course=user_course,
                            user_class=user_class, user_chapter=user_chapter,
                            user_practice=user_practice,
                        )
                        expanded.append(q)
                    return expanded
                time.sleep(1)
            except Exception as inner_e:
                err_str = str(inner_e)
                if "429" in err_str:
                    time.sleep(6)
                elif "500" in err_str or "503" in err_str:
                    time.sleep(3)
                continue

        if last_raw_response:
            try:
                fix_prompt = (
                    "The text below is supposed to be a valid JSON array of question objects "
                    "but has syntax errors. Fix ALL syntax errors and return ONLY the corrected "
                    "JSON array. Do NOT change any content, field names, or values — only fix syntax.\n\n"
                    f"Broken JSON:\n{last_raw_response[:12000]}\n\n"
                    "Return ONLY the valid JSON array. No markdown. No explanation."
                )
                fix_response = model.generate_content(fix_prompt)
                parsed = clean_json_response(fix_response.text)
                if parsed:
                    print(f"✅ Page {page_num}: JSON recovered via fix-prompt ({len(parsed)} questions)")
                    if section_marks_hint:
                        for item in parsed:
                            if not fix_marks_field(item.get("marks", "")):
                                item["marks"] = section_marks_hint
                    if num_diagrams > 0:
                        parsed = auto_inject_missed_diagrams(parsed, page_images)
                    expanded = []
                    for q in parsed:
                        q = clean_question(
                            q, page_images=page_images,
                            user_subject=user_subject, user_course=user_course,
                            user_class=user_class, user_chapter=user_chapter,
                            user_practice=user_practice,
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
# HELPER: derive output JSON filename from uploaded PDF name
# ================================================================
def get_output_json_filename(uploaded_filename: str) -> str:
    """Return '<stem>.json' for the uploaded PDF filename."""
    stem = os.path.splitext(uploaded_filename)[0]
    stem = re.sub(r'[^\w\-.]', '_', stem)
    return f"{stem}.json"

# ================================================================
# MAIN UI
# ================================================================

# Session state
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "extraction_count" not in st.session_state:
    st.session_state.extraction_count = 0
if "extraction_total" not in st.session_state:
    st.session_state.extraction_total = 0
if "extraction_file" not in st.session_state:
    st.session_state.extraction_file = None
if "output_json_filename" not in st.session_state:
    st.session_state.output_json_filename = "questions.json"

st.title("⚡ PDF to JSON - Complete Extraction")
st.markdown("---")

SUBJECT_OPTIONS = ["Hindi", "English", "Math", "Physics", "Chemistry", "Biology", "Social Science"]

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    num_threads = st.slider("Threads (Concurrency)", 1, 5, 2)
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.info("Threads 2-3 recommended for free tier.")
    st.markdown("---")
    st.subheader("📋 Question Metadata")
    user_subject = st.selectbox("Subject Name", options=[""] + SUBJECT_OPTIONS, index=0)
    user_course = st.text_input("Course", placeholder="e.g. JEE Main, NEET, CBSE...")
    user_class = st.text_input("Class", placeholder="e.g. 10, 11, 12...")
    user_chapter = st.text_input("Chapter", placeholder="e.g. Laws of Motion, Algebra...")
    user_practice = st.text_input("Practice", placeholder="e.g. Unit Test-1, Exercise 2.1...")
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

# Clear saved result if a different file is uploaded
if uploaded_file and uploaded_file.name != st.session_state.extraction_file:
    st.session_state.extraction_result = None
    st.session_state.extraction_count = 0
    st.session_state.extraction_total = 0

if any([user_subject, user_course, user_class, user_chapter]):
    with st.expander("📌 Metadata Preview", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Subject", user_subject or "—")
        cols[1].metric("Course", user_course or "—")
        cols[2].metric("Class", user_class or "—")
        cols[3].metric("Chapter", user_chapter or "—")

st.markdown("---")
use_sections = st.toggle("📑 Configure PDF Sections (marks per section)", value=False)
section_configs = []
if use_sections:
    with st.container():
        st.info("Configure section page ranges and marks per question")
        section_configs = render_section_configurator(total_pages_hint=None)

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

    output_json_filename = get_output_json_filename(uploaded_file.name)
    st.session_state.output_json_filename = output_json_filename

    try:
        with st.status("Phase 1: Loading PDF...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            doc = fitz.open(temp_pdf_path)
            total_pages = len(doc)
            doc.close()

            skip_hashes = compute_template_image_hashes(temp_pdf_path)
            if skip_hashes:
                status.update(label=f"Done! {total_pages} pages — {len(skip_hashes)} template image(s) filtered", state="complete")
            else:
                status.update(label=f"Done! {total_pages} pages loaded", state="complete")

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
             page_marks_map[i], skip_hashes)
            for i in pending_indices
        ]

        progress_bar = st.progress(already_done / total_pages if total_pages else 0)
        status_text = st.empty()
        completed = already_done

        if pending_args:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {
                    executor.submit(process_single_page, args): pending_indices[j]
                    for j, args in enumerate(pending_args)
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    result = future.result()
                    page_results[idx] = result if result else []
                    completed += 1
                    save_checkpoint(uploaded_file.name, page_results, total_pages)
                    progress_bar.progress(completed / total_pages)
                    status_text.text(f"Processing page {completed}/{total_pages}...")
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
                recover_status, skip_hashes=skip_hashes
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
            all_questions = [apply_field_order(q) for q in all_questions]

            final_json = json.dumps(all_questions, indent=4, ensure_ascii=False)

            st.session_state.extraction_result = final_json
            st.session_state.extraction_count = final_count
            st.session_state.extraction_total = total_pages
            st.session_state.extraction_file = uploaded_file.name

            st.success(f"✅ Extraction Complete! **{final_count} questions** extracted from **{output_json_filename}**")
            if duplicates_removed:
                st.warning(f"🔁 {duplicates_removed} duplicate(s) auto-removed")
            else:
                st.info("✅ No duplicates found")

            with_py = sum(1 for q in all_questions if str(q.get("previous_year", "")).strip())
            st.info(f"📅 Questions with previous_year filled: **{with_py} / {final_count}**")

            sample_py = list(set(
                str(q.get("previous_year", "")).strip()
                for q in all_questions
                if str(q.get("previous_year", "")).strip()
            ))[:20]
            if sample_py:
                with st.expander(f"📋 Sample previous_year values extracted ({len(sample_py)} unique)"):
                    for v in sorted(sample_py):
                        st.code(v)

            incomplete = [
                q for q in all_questions
                if _question_needs_answer(q) and _s(q.get("question_type")) not in (
                    "Subjective", "Match the Column Question",
                    "Assertion and Reasoning Questions ( A & R )",
                    "Case Based Questions (CBQ)"
                )
            ]
            if incomplete:
                with st.expander(f"⚠️ {len(incomplete)} question(s) still missing Answer/Explanation", expanded=True):
                    for q in incomplete:
                        qid  = q.get("questionid", "?")
                        qtxt = _s(q.get("question"))[:120] + ("..." if len(_s(q.get("question"))) > 120 else "")
                        qtyp = q.get("question_type", "")
                        st.markdown(f"**Q{qid}** `[{qtyp}]` — _{qtxt}_")
            else:
                st.success("✅ All questions complete!")

            with_marks = sum(1 for q in all_questions if str(q.get("marks", "")).strip())
            with_py = sum(1 for q in all_questions if str(q.get("previous_year", "")).strip())
            with_subtopic = sum(1 for q in all_questions if str(q.get("subtopic", "")).strip())
            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Questions with Marks", f"{with_marks} / {len(all_questions)}")
            c2.metric("📅 Questions with Previous Year", f"{with_py} / {len(all_questions)}")
            c3.metric("🏷️ Questions with Subtopic", f"{with_subtopic} / {len(all_questions)}")

            type_counts = {}
            for q in all_questions:
                t = q.get("question_type", "Unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            with st.expander("📊 Question Type Breakdown"):
                for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    st.write(f"• **{q_type}**: {count}")

            with st.expander("Preview (First 5 Questions)"):
                display_questions = []
                for q in all_questions[:5]:
                    display_q = q.copy()
                    text_fields = ["question", "Explanation", "option1", "option2", "option3", "option4"]
                    for field in text_fields:
                        if field in display_q and display_q[field]:
                            display_q[field] = format_text_with_line_breaks(display_q[field])
                    display_questions.append(display_q)

                st.markdown(
                    f'<div style="font-family: monospace; white-space: pre-wrap;">'
                    f'{json.dumps(display_questions, indent=2, ensure_ascii=False)}</div>',
                    unsafe_allow_html=True
                )

            delete_checkpoint(uploaded_file.name)
            st.caption("Checkpoint deleted successfully")

            st.markdown("---")
            if st.button("🔄 Re-extract from Scratch", type="secondary"):
                st.session_state.extraction_result = None
                delete_checkpoint(uploaded_file.name)
                st.rerun()

        else:
            st.error("No questions found. Check if the PDF contains readable text.")
            if st.button("🔄 Retry Extraction", type="primary"):
                st.session_state.extraction_result = None
                delete_checkpoint(uploaded_file.name)
                st.rerun()

    except Exception as global_error:
        st.error(f"Fatal Error: {str(global_error)}")
        st.warning("Progress has been saved. Click 'Start / Resume' to continue.")

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
    output_fname = st.session_state.output_json_filename
    st.success(
        f"📦 Ready to download — **{st.session_state.extraction_count} questions** "
        f"from **{st.session_state.extraction_total} pages** · File: **{output_fname}**"
    )
    st.download_button(
        f"📥 Download {output_fname}",
        data=st.session_state.extraction_result,
        file_name=output_fname,
        mime="application/json",
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
