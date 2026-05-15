import streamlit as st
import google.generativeai as genai
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

st.set_page_config(
    page_title="Ultra-Fast Extraction: PDF to JSON",
    page_icon="⚡",
    layout="wide"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Canonical field order ──
FIELD_ORDER = [
    "questionid", "question", "option1", "option2", "option3", "option4",
    "Answer", "Explanation", "course", "subjectname", "chapter", "practice",
    "subtopic", "medium", "difficulty", "question_type", "previous_year",
    "marks", "class", "book", "question_bucket",
]

# ================================================================
# PyMuPDF PDF HELPERS (no Poppler needed)
# ================================================================

def pdf_page_to_png_bytes(pdf_path: str, page_num: int, dpi: int = 200) -> bytes:
    """Render a single PDF page to PNG bytes using PyMuPDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def compute_template_image_hashes(pdf_path: str) -> set:
    """Return MD5 hashes of images that appear on 3+ pages (logos, headers, template graphics)."""
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
    """Extract all embedded diagram/image objects from a PDF page."""
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
    """Build an HTML img tag for embedding a diagram."""
    return f'<img src="{data_uri}" style="max-width:{width}px;display:block;margin:6px 0"/>'


def inject_diagrams_into_question(q: dict, page_images: list[dict]) -> dict:
    """Embed extracted page images into the correct question field."""
    if not page_images:
        q.pop("diagram_placements", None)
        q.pop("diagram_image_indices", None)
        return q

    diagram_injected = False

    for field in ["question", "Explanation", "option1", "option2", "option3", "option4"]:
        if field in q and q[field]:
            matches = re.findall(r'\[DIAGRAM_(\d+)\]', q[field])
            for match in matches:
                idx = int(match)
                if 0 <= idx < len(page_images):
                    tag = _img_tag(page_images[idx]["data_uri"])
                    q[field] = q[field].replace(f'[DIAGRAM_{idx}]', tag)
                    diagram_injected = True
                else:
                    q[field] = q[field].replace(f'[DIAGRAM_{idx}]', '')

    placements = q.pop("diagram_placements", [])
    indices_fallback = q.pop("diagram_image_indices", [])

    if placements and not diagram_injected:
        for p in placements:
            idx = p.get("image_index", 0)
            field = p.get("field", "Explanation")
            position = p.get("position", "end")
            if 0 <= idx < len(page_images) and field in q:
                tag = _img_tag(page_images[idx]["data_uri"])
                if position == "start":
                    q[field] = tag + "\n" + (q.get(field) or "")
                elif position == "replace" and "[DIAGRAM]" in q.get(field, ""):
                    q[field] = q[field].replace("[DIAGRAM]", tag)
                else:
                    q[field] = (q.get(field) or "") + "\n" + tag
                diagram_injected = True

    if not diagram_injected and indices_fallback:
        question_text = q.get("question", "").lower()
        explanation_text = q.get("Explanation", "").lower()
        # Hindi + English diagram keywords
        diagram_keywords = [
            'figure', 'fig.', 'diagram', 'image', 'shown below', 'given below', 'above figure',
            'चित्र', 'आकृति', 'दिया गया', 'नीचे दिया', 'ऊपर दिया', 'आरेख'
        ]

        for idx in indices_fallback:
            if 0 <= idx < len(page_images):
                if any(keyword in question_text for keyword in diagram_keywords):
                    tag = _img_tag(page_images[idx]["data_uri"], width=300)
                    q["question"] = (q.get("question") or "") + f"\n{tag}"
                    diagram_injected = True
                elif any(keyword in explanation_text for keyword in diagram_keywords):
                    tag = _img_tag(page_images[idx]["data_uri"], width=300)
                    q["Explanation"] = (q.get("Explanation") or "") + f"\n{tag}"
                    diagram_injected = True

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


def newline_to_br(text):
    """Replace all \\n newlines with <br> tags for HTML rendering."""
    if not text or not isinstance(text, str):
        return text
    text = re.sub(r'<br\s*/?>', '\n', text)  # normalise any existing <br> → \n first
    text = text.replace('\n', '<br>')
    return text


def clean_explanation_prefix(text):
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    # English prefixes
    prefix_pattern = re.compile(
        r'^(?:Ans(?:wer)?\.?\s*:?\s*(?:\([A-Da-d]\))?\s*|Sol(?:ution)?\.?\s*:?\s*)',
        re.IGNORECASE
    )
    # Hindi prefixes
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
# SUBTOPIC AUTO-FILL — Hindi + English keywords
# ================================================================

def infer_subtopic_from_question(q: dict, user_subject: str, user_chapter: str) -> str:
    existing = _s(q.get("subtopic"))
    if existing:
        return existing

    question_text = (_s(q.get("question")) + " " + _s(q.get("Explanation"))).lower()
    chapter_lower = user_chapter.lower() if user_chapter else ""
    subject_lower = user_subject.lower() if user_subject else ""

    # Physics (Hindi + English keywords)
    if "physics" in subject_lower or "phy" in subject_lower or "भौतिक" in subject_lower:
        kw_map = [
            (["newton", "force", "friction", "motion", "inertia", "momentum",
              "न्यूटन", "बल", "घर्षण", "गति", "जड़त्व", "संवेग"], "Laws of Motion"),
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
                return subtopic

    # Chemistry (Hindi + English)
    elif "chemistry" in subject_lower or "chem" in subject_lower or "रसायन" in subject_lower:
        kw_map = [
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
                return subtopic

    # Mathematics (Hindi + English)
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
                return subtopic

    # Biology (Hindi + English)
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
                return subtopic

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
        # Hindi continuation words
        'और', 'या', 'लेकिन', 'इसलिए', 'अतः', 'यदि', 'जब', 'जहाँ', 'जो',
        'यह', 'वह', 'इस', 'उस', 'से', 'के', 'की', 'को', 'में', 'पर'
    ]
    first_word = next_start.split()[0].lower() if next_start.split() else ""
    ends_mid = prev_end[-1] not in '.!?।'  # Added Hindi danda
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
    """Safe strip — handles None/non-string values."""
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
        return "Assertion and Reasoning Questions ( A & R )"
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
                   user_chapter="", user_practice="", user_book="",
                   is_hindi=False):
    """Full post-processing pipeline for a single extracted question."""
    math_fields     = ["question", "option1", "option2", "option3", "option4", "Explanation"]
    metadata_fields = ["subjectname", "chapter", "practice", "subtopic", "medium",
                       "difficulty", "question_type", "course"]

    for k in list(q.keys()):
        if q[k] is None:
            q[k] = ""

    q.pop("category", None)
    q["question_type"] = normalize_question_type(q.get("question_type") or "MCQs")
    q["question_bucket"] = normalize_question_bucket(q.get("question_bucket") or "")

    # Set medium based on Hindi flag
    q["medium"] = "English" if is_hindi else q.get("medium", "English")

    # ── FIX: Extract previous_year from question text ──
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
    if user_book:     q["book"]        = user_book

    # Force medium override
    q["medium"] = "Hindi" if is_hindi else q.get("medium", "English")

    if not _s(q.get("subtopic")):
        inferred = infer_subtopic_from_question(q, user_subject, user_chapter)
        if inferred:
            q["subtopic"] = inferred

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
                                  user_chapter, user_practice, user_book,
                                  skip_hashes=None, is_hindi=False):
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

        lang_instruction = (
            "LANGUAGE: This is a Hindi medium PDF. Extract ALL text in Hindi exactly as it appears. "
            "Do NOT translate to English. Preserve all Devanagari script.\n\n"
            if is_hindi else ""
        )

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
7. IMPORTANT — previous_year: Look for references like (Example-1), (Exercise-7.1-8), JEE Main 2024, etc.

Return ONLY this JSON:
{{
  "Answer": "<1/2/3/4 or 1/0 for True/False or numeric, or empty if not found>",
  "Explanation": "<exact verbatim solution text from the image, every step>",
  "question": "<verbatim continuation of question text if it continues here, else empty>",
  "option1": "<verbatim continuation of option A if split, else empty>",
  "option2": "<verbatim continuation of option B if split, else empty>",
  "option3": "<verbatim continuation of option C if split, else empty>",
  "option4": "<verbatim continuation of option D if split, else empty>",
  "previous_year": "<extracted reference or empty>"
}}

If nothing related to this question appears on this page, return: {{}}
"""

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
                        "course": "", "subjectname": "", "chapter": "", "practice": "",
                        "subtopic": "", "medium": "Hindi" if is_hindi else "English",
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

    diagram_kws = [
        'figure', 'fig.', 'fig ', 'diagram', 'image', 'shown', 'given', 'above', 'below',
        'graph', 'circuit', 'structure',
        'चित्र', 'आकृति', 'दिया गया', 'नीचे', 'ऊपर', 'आरेख', 'ग्राफ', 'परिपथ'
    ]
    for idx in missing:
        tag = f'[DIAGRAM_{idx}]'
        injected = False
        for q in questions:
            q_text = (q.get("question", "") or "").lower()
            if any(kw in q_text for kw in diagram_kws) and tag not in (q.get("question", "") or ""):
                q["question"] = (q.get("question", "") or "") + f"\n{tag}"
                injected = True
                break
        if not injected:
            if questions:
                last = questions[-1]
                last["Explanation"] = (last.get("Explanation", "") or "") + f"\n{tag}"

    return questions


# ================================================================
# ENHANCED GEMINI PAGE PROCESSOR — HINDI AWARE
# ================================================================
def process_single_page(args):
    (pdf_path, page_index, page_num, api_key, model_name,
     user_subject, user_course, user_class, user_chapter, user_practice,
     user_book, section_marks_hint, skip_hashes, is_hindi) = args
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
                f'  They are provided as separate images after the page image below.\n'
                f'  MANDATORY: Place [DIAGRAM_X] (X = index) EXACTLY at the position in the text where the diagram appears.\n'
                f'  • If the diagram is part of the question stem → put [DIAGRAM_X] inside "question"\n'
                f'  • If the diagram is part of an option → put [DIAGRAM_X] inside that option field\n'
                f'  • If the diagram is in the explanation → put [DIAGRAM_X] inside "Explanation"\n'
                f'  • If the diagram appears ABOVE the question → put [DIAGRAM_X] at the START of "question"\n'
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

        subject_ctx = f'Subject: "{user_subject}"' if user_subject else ""
        chapter_ctx = f'Chapter: "{user_chapter}"' if user_chapter else ""
        practice_ctx = f'Practice/Exercise: "{user_practice}"' if user_practice else ""
        context_block = " | ".join(filter(None, [subject_ctx, chapter_ctx, practice_ctx]))

        # ── HINDI-SPECIFIC LANGUAGE BLOCK ──
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
6. Only fixed system fields stay in English: "medium", "difficulty", "question_type",
   "question_bucket" — these use their standard English values as defined below.
7. If any mixed-language content exists, keep exactly as printed.
"""
        else:
            language_block = ""

        prompt = f"""
You are a precise question paper digitizer. Extract EVERY question from this page into a JSON array.

{marks_hint_text}

{language_block}

{'CONTEXT: ' + context_block if context_block else ''}

══════════════════════════════════════════════
RULE 1 — COMPLETENESS (MOST IMPORTANT)
══════════════════════════════════════════════
• Copy ALL text VERBATIM — do NOT paraphrase, summarize, or abbreviate anything.
• Extract EVERY question on this page — do not skip even one.
• Extract ALL option text completely — even if options are long paragraphs.
• Extract the FULL explanation/solution — every step, every line.
• If a table appears inside a question or option, transcribe it as plain text rows separated by \\n.
• If a question has sub-parts (i), (ii), (iii) — include ALL sub-parts in "question".

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
RULE 3 — DIAGRAMS
══════════════════════════════════════════════
{diagram_note}

══════════════════════════════════════════════
RULE 4 — LATEX
══════════════════════════════════════════════
• Inline math  → \\\\( ... \\\\)
• Display math → \\\\[ ... \\\\]
• NEVER use $...$ or $$...$
• Convert ALL mathematical expressions, fractions, superscripts, subscripts to LaTeX.

══════════════════════════════════════════════
RULE 5 — FIELD VALUES
══════════════════════════════════════════════
Answer:
  • MCQs      → "1" / "2" / "3" / "4"  (A=1, B=2, C=3, D=4)
  • True/False → "1" for True/सत्य, "0" for False/असत्य
  • Numeric    → numeric string e.g. "42" or "3.14"
  • Others     → ""

Marks: single digit string only ("1","2","3","4","5") or "" if not mentioned.

medium: ALWAYS use "Hindi" for this extraction.

question_type: exactly one of:
  MCQs | True/False | Numeric | Subjective | Filling Blank |
  Assertion and Reasoning Questions ( A & R ) |
  Match the Column Question | Case Based Questions (CBQ)

difficulty: Easy / Medium / Hard (your assessment)

question_bucket: carefully assess each question and assign exactly one of:
  Beginner | Target | Advance Climb | Must Do

══════════════════════════════════════════════
RULE 6 — SUBTOPIC (VERY IMPORTANT — DO NOT LEAVE BLANK)
══════════════════════════════════════════════
"subtopic" must ALWAYS be filled with a meaningful value.
  • Read the question carefully and identify the SPECIFIC concept/topic being tested.
  • Write the subtopic in English (standard subject terminology).
  • Examples: "Newton's Laws", "Integration by Parts", "Mole Concept", "Photosynthesis" etc.
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

══════════════════════════════════════════════
OUTPUT FORMAT — repeat for EVERY question:
══════════════════════════════════════════════
{{
  "questionid": "<number as printed, or empty for continuation fragment>",
  "question": "<complete verbatim question text{' in Hindi' if is_hindi else ''}, with LaTeX and [DIAGRAM_X] if needed — WITHOUT any reference tag>",
  "option1": "<option A full text{' in Hindi' if is_hindi else ''}, no prefix>",
  "option2": "<option B full text{' in Hindi' if is_hindi else ''}, no prefix>",
  "option3": "<option C full text{' in Hindi' if is_hindi else ''}, no prefix>",
  "option4": "<option D full text{' in Hindi' if is_hindi else ''}, no prefix>",
  "Answer": "<per rules above>",
  "Explanation": "<complete solution — every line verbatim{' in Hindi' if is_hindi else ''}>",
  "course": "",
  "subjectname": "",
  "chapter": "",
  "practice": "",
  "subtopic": "<specific concept/topic being tested — NEVER leave blank — in English>",
  "medium": "Hindi",
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
                            user_practice=user_practice, user_book=user_book,
                            is_hindi=is_hindi
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

        # All 3 attempts failed — ask Gemini to fix broken JSON
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

st.title("⚡ PDF to JSON - Complete Extraction")
st.markdown("---")

SUBJECT_OPTIONS = ["Hindi", "English", "Math", "Physics", "Chemistry", "Biology", "Social Science",
                   "हिंदी", "अंग्रेज़ी", "गणित", "भौतिक विज्ञान", "रसायन विज्ञान", "जीव विज्ञान", "सामाजिक विज्ञान"]

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    num_threads = st.slider("Threads (Concurrency)", 1, 5, 2)
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.info("Threads 2-3 recommended for free tier.")
    st.markdown("---")

    # ── HINDI MEDIUM TOGGLE ──
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
        # NOTE: Added is_hindi to args tuple (14 elements now)
        pending_args = [
            (temp_pdf_path, i, i + 1, api_key, model_choice,
             user_subject, user_course, user_class, user_chapter, user_practice,
             user_book, page_marks_map[i], skip_hashes, is_hindi)
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
                # Final force: ensure medium is correct in every question
                q["medium"] = "Hindi" if is_hindi else q.get("medium", "English")

            all_questions = [apply_field_order(q) for q in all_questions]

            # ══════════════════════════════════════════
            # JSON DUMP — ensure_ascii=False so Hindi
            # characters render correctly on all systems
            # ══════════════════════════════════════════
            final_json = json.dumps(all_questions, indent=4, ensure_ascii=False)

            st.session_state.extraction_result = final_json
            st.session_state.extraction_count = final_count
            st.session_state.extraction_total = total_pages
            st.session_state.extraction_file = uploaded_file.name

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
                    "Assertion and Reasoning Questions ( A & R )",
                    "Case Based Questions (CBQ)"
                )
            ]
            subj_incomplete = [
                q for q in all_questions
                if _s(q.get("question_type")) in (
                    "Subjective", "Match the Column Question",
                    "Assertion and Reasoning Questions ( A & R )",
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
