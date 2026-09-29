"""Generate the parity fixtures for ``frontend/scripts/verify-text2voxel.mjs``.

The browser engine re-implements, in TypeScript, several things the Python side
does with libraries: the HF ``tokenizers`` BERT pipeline, the MiniLM mean-pooled
embedding (``training/text2voxel/common.py`` ``TextEncoder``), numpy's DDIM
timestep schedule, the backend's dimension parser and its blueprint sheet. This
script records what the Python reference produces so the Node verification can
compare the shipped TypeScript against it.

It also writes ``frontend/src/lib/text2voxel/bertTables.ts``: the character
classes the HF normaliser / pre-tokenizer use (control characters, CJK ranges,
punctuation, non-spacing marks). They are *probed* from the installed
``tokenizers`` build rather than taken from the browser's Unicode tables,
because the Rust crate's tables do not match any one Unicode version exactly
(e.g. its CJK range starts at U+2B920, not U+2B820 as in Google's BERT).

Run with the backend venv (needs tokenizers, onnxruntime, numpy, pydantic):

    C:/Users/arunk/.venvs/ai3d-blueprint/Scripts/python.exe \
        frontend/scripts/fixtures/make_text2voxel_fixtures.py

Inputs: ``frontend/public/models/minilm/{tokenizer.json,model_quantized.onnx}``.
Runtime: ~15 s (the code-point probes dominate), a few dozen MiniLM calls.
"""

from __future__ import annotations

import base64
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRONTEND = HERE.parent.parent
REPO = FRONTEND.parent
MINILM = FRONTEND / "public" / "models" / "minilm"
FIXTURE_OUT = HERE / "text2voxel_fixtures.json"
PROBE_OUT = HERE / "unicode_probe.json"
TABLES_OUT = FRONTEND / "src" / "lib" / "text2voxel" / "bertTables.ts"
SAMPLER_OUT = HERE / "sampler_reference.json"
MODEL = FRONTEND / "public" / "models" / "text2voxel"

sys.path.insert(0, str(REPO / "training" / "text2voxel"))
sys.path.insert(0, str(REPO / "backend"))

from common import (  # noqa: E402
    MAX_TOKENS,
    MODELNET_NAMES,
    PALETTE,
    TextEncoder,
    all_template_strings,
)

MAX_CODE_POINT = 0x110000


def is_surrogate(cp: int) -> bool:
    return 0xD800 <= cp <= 0xDFFF


# --------------------------------------------------------------------------
# Test strings
# --------------------------------------------------------------------------

#: Hand-written captions in the style of Text2Shape (lower-effort human text,
#: typos and all) plus the kind of prompt a user types into the web UI.
HAND_CAPTIONS = [
    "a red office chair with armrests and wheels",
    "This is a brown wooden chair with four legs and a slatted back.",
    "A dark brown wooden dining table with a rectangular top and four thick legs.",
    "grey fabric sofa, three seats, low wooden legs",
    "a modern minimalist oak dining chair with tapered legs",
    "a round glass coffee table with metal legs",
    "This chair has a high back and curved armrests. The seat is padded and black.",
    "a white plastic stackable chair",
    "The table is made of light colored wood and has a drawer underneath the top.",
    "Mid-century modern lounge chair (Eames-style) with a walnut shell & black leather cushions.",
    "a 3-seater sofa, 2.1m long, in navy blue velvet!",
    "A futuristic silver sports car with a low roof and big rear spoiler",
    "an old red pickup truck with rusty wheels",
    "a yellow school bus",
    "a sleek black motorcycle with chrome exhaust pipes",
    "a small white sailboat with a tall mast",
    "a green tractor with huge rear wheels",
    "a blue passenger jet with two engines under the wings",
    "a military helicopter, olive drab, with a long tail boom",
    "A 4-door sedan; metallic grey paint; tinted windows.",
    "its a chair.. brown.. with 4 legs and no arms",
    "wooden stool w/ round seat & 3 legs",
    "Rectangular table, black top, silver legs -- very simple design",
    "a bar stool that is tall and has a footrest ring",
    "the chair is blue and has wheels at the bottom of its 5 legs",
    "A tall bookshelf with 5 shelves (pine wood)",
    "a king-size bed with a padded headboard",
    "a nightstand w/ two drawers + brass handles",
    "Swivel office chair: mesh back, adjustable height, lumbar support",
    "square coffee table, 90 x 90 x 45 cm, oak veneer",
    "a desk lamp with an adjustable arm",
    "an Adirondack chair painted white",
    "rocking chair made out of dark stained wood with spindle back",
    "A glass-topped dining table seating 6-8 people",
    "a folding metal chair (grey)",
    "a table that looks like a tree stump",
    "L-shaped corner sofa in light grey fabric with chaise",
    "a red sports car, Ferrari-like, 2 seats",
    "a vintage VW-style camper van, two-tone blue/white",
    "an F1 race car with front and rear wings",
    "a cargo ship with containers stacked on deck",
    "a fire truck with a ladder on top",
    "a pink bicycle with a basket on the front",
    "A pale yellow armchair with rolled arms and skirted base",
    "this is a tall chair. it's legs are thin and made of metal. the seat is wooden",
    "Chair: black; legs: 4; arms: none; back: tall & straight.",
    "a dining chair w. a woven rattan seat",
    "A round pedestal table with a single central column and a 3-footed base",
    "sofa bed that folds out",
    "a simple wooden bench without back",
    "an ottoman / footstool in brown leather",
    "a tv stand with open shelves and glass doors",
    "a coffee table with a lower shelf for magazines",
    "a double-decker bus (red, London style)",
    "an SUV with roof rails and alloy wheels",
    "a delivery van, white, with sliding side door",
    "a steam locomotive with a tall chimney",
    "a propeller airplane with high wings",
    "a speedboat - white hull - blue stripe",
    "a tank with a long cannon and caterpillar tracks",
]

OBJECTS = [
    "chair", "armchair", "office chair", "dining chair", "bar stool", "stool", "table",
    "coffee table", "dining table", "side table", "desk", "sofa", "couch", "loveseat",
    "bench", "bed", "bookshelf", "cabinet", "dresser", "nightstand", "lamp", "car",
    "sports car", "pickup truck", "suv", "van", "bus", "motorcycle", "bicycle", "airplane",
    "jet", "helicopter", "boat", "sailboat", "tractor", "train", "truck", "limousine",
]
COLOURS = [
    "red", "dark brown", "light grey", "navy blue", "black", "white", "beige", "green",
    "yellow", "orange", "purple", "pink", "silver", "tan", "maroon", "teal", "cream",
    "charcoal", "light blue", "dark green",
]
MATERIALS = [
    "wooden", "oak", "walnut", "metal", "steel", "chrome", "leather", "fabric", "plastic",
    "glass", "rattan", "velvet", "aluminium", "bamboo", "marble",
]
FEATURES = [
    "with four legs", "with armrests", "with wheels", "with a curved back",
    "with a glass top", "with two drawers", "with spoked wheels", "with a rear spoiler",
    "with tinted windows", "with a high back", "with thin tapered legs",
    "with a padded seat", "with a cushion", "without arms", "with a slatted back",
    "with a round base", "with a sunroof", "with big tyres", "with a single pedestal",
    "with cross bars between the legs",
]
TEMPLATES = [
    "a {c} {m} {o} {f}",
    "This is a {c} {o}. It is made of {m2} and {f2}.",
    "{C} {o} {f}",
    "a {m} {o} that is {c}",
    "The {o} is {c}; it has {f3}.",
    "{c} {o}, {m}, {f}",
    "an old {c} {o} {f}!",
    "A {m} {o} ({c}) {f}.",
]


def generated_captions(count: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        template = rng.choice(TEMPLATES)
        colour = rng.choice(COLOURS)
        material = rng.choice(MATERIALS)
        feature = rng.choice(FEATURES)
        text = template.format(
            c=colour, C=colour.capitalize(), m=material, m2=material.replace("en", ""),
            o=rng.choice(OBJECTS), f=feature, f2=feature.replace("with", "has"),
            f3=feature.replace("with ", "").replace("without", "no"),
        )
        text = text.replace(" a o", " an o").replace(" a a", " an a")
        if text not in out:
            out.append(text)
    return out


EDGE_CASES = [
    "",
    " ",
    "   \t\n  ",
    "a",
    "A",
    "chair",
    "CHAIR",
    "ChAiR!!!",
    "chair...",
    "chair?!",
    "(chair)",
    "[chair]",
    "{table}",
    "<sofa>",
    "a\"quoted\" chair",
    "the chair's leg",
    "don't",
    "hello,world",
    "one-two-three",
    "snake_case_name",
    "path/to/file.glb",
    "C:\\models\\chair.obj",
    "email@example.com",
    "https://example.com/chair?id=42&x=y#top",
    "$100 + 20% = ~$120 ^ | ` ~",
    "50 x 30 x 20 cm",
    "50cm x 30cm x 20cm",
    "2 x 1.5 x 0.8 m",
    "1.2 m tall",
    "height of 40 cm",
    "3/4\" plywood, 6'2\" tall",
    "1,234,567.89",
    "007 bond car",
    "0.5mm",
    "1e6 3.14159 -42 +7",
    "#hashtag @mention",
    "a[SEP]b",
    "[CLS] chair [SEP]",
    "[MASK] table",
    "[UNK]",
    "[cls] lowercase is not special",
    "[PAD][PAD]",
    "a" * 99,
    "a" * 100,
    "a" * 101,
    "chair " + "b" * 150,
    "supercalifragilisticexpialidocious",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "antidisestablishmentarianism table",
    " ".join(["chair"] * 140),
    " ".join(f"word{i}" for i in range(80)),
    ", ".join(["a red chair with arms"] * 20),
]

UNICODE_CASES = [
    "café chair",
    "CAFÉ",
    "naïve résumé",
    "Ångström",
    "Zürich straße",
    "GROẞE STRAẞE",
    "São Paulo sofá",
    "crème brûlée table",
    "e\u0301 combining acute",
    "\u0301 lone combining mark",
    "a\u0308\u0304 stacked marks",
    "ΟΔΟΣ Σίσυφος",
    "İstanbul ıi",
    "Москва стул",
    "Київ ґ",
    "中文椅子",
    "我想要一把红色的椅子",
    "日本語のテーブル",
    "カタカナ ｶﾀｶﾅ",
    "한국어 의자",
    "한글",
    "ＡＢＣ　ｆｕｌｌｗｉｄｔｈ",
    "ﬁne ﬂoor",
    "𝐀𝐁𝐂 math bold",
    "x² + y³ = ½",
    "Ⅳ Ⅻ roman",
    "مرحبا كرسي",
    "שלום כיסא",
    "สวัสดี เก้าอี้",
    "नमस्ते कुर्सी",
    "Ꭰ Cherokee Ꮳ",
    "ᲐᲑᲒ Mtavruli",
    "a chair 🪑",
    "🛋️ sofa",
    "🚗🚕🚙",
    "car🚗",
    "👍🏽 thumbs",
    "👨‍👩‍👧 family",
    "🇬🇧 flag",
    "❤️ love it",
    "★☆✓✗",
    "€100 £50 ¥1000 ₹",
    "©®™",
    "…—–‘’“”",
    "«quotes» ‹single›",
    "¿Qué? ¡Sí!",
    "zero\u200bwidth",
    "joiner\u200dhere",
    "bom\ufeffinside",
    "nbsp\u00a0space",
    "ideographic\u3000space",
    "nel\u0085line",
    "tab\there",
    "line\nbreak",
    "carriage\rreturn",
    "null\x00byte",
    "bell\x07char",
    "replacement\ufffdchar",
    "private\ue000use",
    "\U00020000 ext B",
    "\U0002b820 ext E start",
    "\U0002b920 hf range",
    "\U0002ebf0 ext I",
    "\U00030000 ext G",
    "⼀ kangxi",
    "㐀 ext A",
    "豈 compat",
    "Hello\u2028World",
    "variation\ufe0fselector",
]


def random_strings(count: int, seed: int = 11) -> list[str]:
    """Random mixes of ASCII words and code points from assigned blocks."""
    rng = random.Random(seed)
    blocks = [
        (0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
        (0x590, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF), (0x1E00, 0x1FFF),
        (0x2000, 0x206F), (0x2070, 0x209F), (0x20A0, 0x20CF), (0x2100, 0x218F),
        (0x2190, 0x23FF), (0x2460, 0x27BF), (0x2E00, 0x2E7F), (0x3000, 0x30FF),
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0xF900, 0xFAFF),
        (0xFE00, 0xFE6F), (0xFF00, 0xFFEF), (0x1D400, 0x1D7FF), (0x1F300, 0x1FAFF),
        (0x20000, 0x2A6DF), (0x2B740, 0x2CEAF), (0xE0000, 0xE007F),
    ]
    words = ["chair", "table", "car", "sofa", "red", "wooden", "with", "legs", "a", "the"]
    out = []
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(1, 8)):
            if rng.random() < 0.4:
                parts.append(rng.choice(words))
            else:
                low, high = rng.choice(blocks)
                cp = rng.randint(low, high)
                if is_surrogate(cp):
                    cp = 0x41
                glue = rng.choice(["", " ", "", "-"])
                parts.append(glue + chr(cp) * rng.randint(1, 3))
        out.append(rng.choice([" ", "", ", "]).join(parts))
    return out


TEST_PROMPTS = [
    "a modern minimalist oak dining chair with tapered legs",
    "a red office chair with armrests and wheels",
    "a round glass coffee table with metal legs",
    "a long wooden dining table",
    "a blue sofa",
    "a futuristic silver sports car",
    "an airplane",
    "a floor lamp",
    "a toilet",
    "a guitar",
    "a computer monitor",
    "a bed",
    "a wooden bookshelf",
    "a flower pot",
    "a bench",
    "a green bathtub",
]


def template_sample(count: int, seed: int = 3) -> list[str]:
    strings = all_template_strings()
    rng = random.Random(seed)
    return rng.sample(strings, count)


# --------------------------------------------------------------------------
# Dimension parser cases (backend/app/pipeline/prompt_parser.py)
# --------------------------------------------------------------------------

DIMENSION_PROMPTS = [
    "a coffee table 120 x 60 x 45 cm",
    "a coffee table 120x60x45cm",
    "a box 50cm x 30cm x 20cm",
    "shelf 2 x 0.3 x 1.8 m",
    "a desk 1.4 X 0.7 X 0.75 m",
    "cabinet 800 × 400 × 900 mm",
    "a tray 12 x 8 x 1 in",
    "a crate 3 x 2 x 2 ft",
    "table 120 x 60 cm x 75",
    "table 120 x 60 x 75",
    "a chair 45cm x 50 x 90 cm",
    "a 1.2 m tall floor lamp",
    "a lamp 1.2m tall",
    "a stool with a height of 65 cm",
    "a stool, height: 65cm",
    "a stool height=65 cm",
    "a bench 40cm wide and 1.8 m long",
    "a sofa 2.1 metres long, 90 cm deep, 85cm high",
    "a bookcase 30 inches wide",
    "a tower 6 feet tall",
    "a door 7 ft in height",
    "a 40 min long walk",
    "a table 120 x 80 x 75 cm and 45 cm tall",
    "a chair",
    "a red car",
    "a car 4.5 m long",
    "a car length of 4.5 m and width 1.8 m",
    "an airplane with a width of 30 m",
    "a vase 30 centimetres tall",
    "a vase 30 centimeters high",
    "a table 1200 millimetres long",
    "a table 1200 millimeters long",
    "a bed 2 metre long",
    "a bed 2 meters long 1.6 meters wide",
    "a mug 10.5 cm tall",
    "a plank 2.4m x 0.2m x 0.05m",
    "12 X 12 X 12 IN cube",
    "a lamp 150cm TALL",
    "size: 10 x 20",
    "dimensions 10 x 20 x 30 furlongs",
    "a monitor 60 cm across",
    "a pool 10 m long 5 m wide 2 m deep",
    "3.5 x 2.25 x 1.125 m platform",
    "a shelf 0.0625 m tall",
    "a box 1.0625 x 2.0625 x 3.0625 in",
    "a table 33.3333 cm tall",
    "   a    table    120    x   60   x   45   cm   ",
    "",
    "ab",
    "a chair 45 cm wide, 45 cm deep and 90 cm tall",
    "a desk (160 x 80 x 74 cm) in oak",
    "a desk 160 by 80 by 74 cm",
    "a table 2m long, a chair 1m tall",
    "a wardrobe with a height of 2.1 m and a width of 1.2 m",
    "a 42 inch wide tv stand",
    "a 5 ft long bench",
    "width: 70cm, depth: 60cm, height: 110cm",
    "a sofa of length 220cm",
]


def dimension_cases() -> list[dict]:
    from app.pipeline.prompt_parser import _extract_dimensions  # noqa: PLC0415

    cases = []
    for prompt in DIMENSION_PROMPTS:
        text = re.sub(r"\s+", " ", prompt.strip())
        dims = _extract_dimensions(text)
        cases.append({
            "prompt": prompt,
            "length": dims.length_mm,
            "width": dims.width_mm,
            "height": dims.height_mm,
        })
    return cases


# --------------------------------------------------------------------------
# Blueprint sheet cases (backend/app/pipeline/blueprint_sheet.py)
# --------------------------------------------------------------------------


def sheet_cases() -> list[dict]:
    from app.pipeline.blueprint_sheet import render_blueprint_sheet  # noqa: PLC0415

    rng = random.Random(5)

    def polyline(n: int, w: float, h: float, x0: float, y0: float) -> list:
        return [[x0 + rng.random() * w, y0 + rng.random() * h] for _ in range(n)]

    def view(w: float, h: float, x0: float, y0: float, empty: bool = False) -> dict:
        if empty:
            return {"outline": [], "inline": [], "hidden": [], "width": 0.0, "height": 0.0}
        outline = [polyline(rng.randint(4, 12), w, h, x0, y0) for _ in range(rng.randint(1, 3))]
        # Anchor the extents so minimum_x/minimum_y are exactly x0/y0.
        outline[0][0] = [x0, y0]
        outline[0][1] = [x0 + w, y0 + h]
        return {
            "outline": outline,
            "inline": [polyline(rng.randint(2, 6), w, h, x0, y0) for _ in range(rng.randint(0, 4))],
            "hidden": [polyline(rng.randint(2, 5), w, h, x0, y0) for _ in range(rng.randint(0, 3))],
            "width": w,
            "height": h,
        }

    cases = []
    specs = [
        None,
        {"materials": ["oak", "steel"], "tolerances_mm": 0.5},
        {"materials": [], "tolerances_mm": None},
        {"materials": ["glass <tempered> & \"clear\""], "tolerances_mm": 2.0},
    ]
    sizes = [
        (0.46, 0.92, 0.51),     # chair-ish
        (1.2, 0.75, 0.6),       # table
        (4.5, 1.4, 1.9),        # car
        (0.0125, 0.0625, 0.03125),  # tiny: exercises the .1f / tie branches
        (0.0055, 0.0045, 0.0105),
        (12.5, 3.5, 2.25),
    ]
    names = [
        "Generated model",
        "A red office chair with armrests and wheels, extra long name here",
        "Chair <script>alert('x')</script> & \"quotes\"",
        "Ünïcödé 椅子 🪑 name",
        "",
    ]
    index = 0
    for w, h, d in sizes:
        for theme in ("white", "blueprint"):
            drawing = {
                "front": view(w, h, -w / 2, 0.0),
                "side": view(d, h, -d / 2, 0.0),
                "top": view(w, d, -w / 2, -d / 2),
            }
            if index % 7 == 5:
                drawing["side"] = view(0, 0, 0, 0, empty=True)
            stats = {"triangles": rng.choice([0, 12, 1234, 98765, 1234567])}
            spec = specs[index % len(specs)]
            name = names[index % len(names)]
            svg = render_blueprint_sheet(drawing, stats, spec, name=name, theme_name=theme)
            cases.append({"drawing": drawing, "stats": stats, "spec": spec, "name": name,
                          "theme": theme, "svg": svg})
            index += 1
    return cases


# --------------------------------------------------------------------------
# Unicode probes -> shipped tables + a full-coverage fixture
# --------------------------------------------------------------------------


def to_ranges(code_points: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for cp in sorted(code_points):
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])
    return ranges


#: Code points per digest block in the full-coverage normaliser fixture.
PROBE_BLOCK = 4096


def block_digest(outputs: list[str]) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256("\x01".join(outputs).encode("utf-8")).hexdigest()[:16]


def probe_unicode(tokenizer) -> dict:
    """Classify every code point the way the HF normaliser / pre-tokenizer does.

    Returns the range tables the browser ships with, plus a digest per block of
    ``PROBE_BLOCK`` code points of the *complete* per-character normalisation
    (clean text, CJK padding, NFD, mark stripping, lowercasing). The browser
    side uses its own NFD and lowercasing, so the digests are what prove those
    agree with the Rust implementation over all 1.1M code points.
    """
    from tokenizers import normalizers  # noqa: PLC0415

    full = tokenizer.normalizer
    clean_only = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False)
    chinese_only = normalizers.BertNormalizer(
        clean_text=False, handle_chinese_chars=True, strip_accents=False, lowercase=False)
    strip_only = normalizers.BertNormalizer(
        clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=False)
    # The same NFD implementation strip_accents runs; its Unicode tables are
    # older than current browsers', so only these code points are decomposed.
    nfd_only = normalizers.NFD()
    pre = tokenizer.pre_tokenizer

    removed, whitespace, chinese, marks, punct, split_space, decompose = [], [], [], [], [], [], []
    digests: list[str] = []
    outputs: list[str] = []
    for cp in range(MAX_CODE_POINT):
        if cp % PROBE_BLOCK == 0 and cp:
            digests.append(block_digest(outputs))
            outputs = []
        if is_surrogate(cp):
            outputs.append("")  # keeps blocks aligned; the browser skips them the same way
            continue
        ch = chr(cp)
        cleaned = clean_only.normalize_str(ch)
        if cleaned == "":
            removed.append(cp)
        elif cleaned == " " and cp != 0x20:
            whitespace.append(cp)
        if chinese_only.normalize_str(ch) == f" {ch} ":
            chinese.append(cp)
        if strip_only.normalize_str(ch) == "":
            marks.append(cp)
        if nfd_only.normalize_str(ch) != ch:
            decompose.append(cp)
        outputs.append(full.normalize_str(ch))
        pieces = [p[0] for p in pre.pre_tokenize_str(f"a{ch}a")]
        if pieces == ["a", ch, "a"]:
            punct.append(cp)
        elif pieces == ["a", "a"]:
            split_space.append(cp)
    digests.append(block_digest(outputs))

    return {
        "removed": to_ranges(removed),
        "whitespace": whitespace,
        "chinese": to_ranges(chinese),
        "marks": to_ranges(marks),
        "punct": to_ranges(punct),
        "split_space": split_space,
        "decompose": to_ranges(decompose),
        "block": PROBE_BLOCK,
        "normalized_digests": digests,
    }


def write_tables(probe: dict) -> None:
    def encode(ranges: list[list[int]]) -> str:
        # Delta-encoded base-36 pairs: gap from the previous end, then length.
        parts, previous = [], 0
        for start, end in ranges:
            parts.append(f"{np.base_repr(start - previous, 36).lower()}"
                         f".{np.base_repr(end - start, 36).lower()}")
            previous = end
        return ",".join(parts)

    body = f'''// GENERATED by frontend/scripts/fixtures/make_text2voxel_fixtures.py - do not edit.
//
// Character classes probed from the HF `tokenizers` build the training data was
// tokenised with, so the browser tokenizer does not depend on the browser's own
// Unicode tables. Each table is a list of inclusive code-point ranges, stored as
// base-36 "gap.length" pairs (gap from the previous range's end).

/** Removed by BertNormalizer clean_text (NUL, U+FFFD, control/format/private-use). */
export const REMOVED = '{encode(probe["removed"])}'

/** Mapped to a plain space by clean_text; also split points for the pre-tokenizer. */
export const WHITESPACE = '{encode(to_ranges(probe["whitespace"]))}'

/** Padded with spaces by handle_chinese_chars (note: HF starts Ext-E at U+2B920). */
export const CHINESE = '{encode(probe["chinese"])}'

/** Non-spacing marks dropped by strip_accents after NFD. */
export const MARKS = '{encode(probe["marks"])}'

/** Isolated as single-character tokens by BertPreTokenizer (ASCII punct + Unicode P*). */
export const PUNCTUATION = '{encode(probe["punct"])}'

/** Split points for BertPreTokenizer (Rust char::is_whitespace). */
export const SPLIT_SPACE = '{encode(to_ranges(probe["split_space"]))}'

/**
 * Code points the Rust NFD decomposes. Browsers ship newer Unicode tables
 * that also decompose characters added since, so NFD is applied only to these.
 */
export const DECOMPOSE = '{encode(probe["decompose"])}'
'''
    TABLES_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLES_OUT.write_text(body, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def b64_floats(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def write_sampler_reference() -> None:
    """Exact-condition sampler reference for the installed model.

    ``meta.reference`` was computed on the training machine, and the int8
    MiniLM model gives slightly different embeddings on different CPUs
    (measured: 8e-3 per component between Kaggle and a Windows laptop, which
    50 guided DDIM steps amplify to ~8e-2 in the latent). Recording the
    condition this machine produces lets the Node check feed the sampler the
    identical vector and hold it to a tight tolerance. Keyed by the prior's
    sha256 so a new model forces regeneration.
    """
    import hashlib  # noqa: PLC0415
    import math  # noqa: PLC0415

    import onnxruntime as ort  # noqa: PLC0415

    meta = json.loads((MODEL / "meta.json").read_text(encoding="utf-8"))
    ref, diffusion = meta["reference"], meta["diffusion"]
    cond = TextEncoder(MINILM).encode([ref["prompt"]], batch_size=1)[0]
    session = ort.InferenceSession(str(MODEL / "prior.onnx"), providers=["CPUExecutionProvider"])
    alphas = np.asarray(diffusion["alphas_cumprod"], dtype=np.float64)
    times = np.linspace(999, 0, diffusion["sample_steps"]).round().astype(int)
    guidance = float(diffusion["guidance"])
    x = np.asarray(ref["noise"], dtype=np.float32)[None]
    for i, t in enumerate(times):
        a = float(alphas[t])
        a_prev = float(alphas[times[i + 1]]) if i + 1 < len(times) else 1.0
        v = session.run(None, {
            "x": np.concatenate([x, x]), "t": np.full(2, float(t), dtype=np.float32),
            "cond": np.stack([cond, np.zeros_like(cond)]).astype(np.float32)})[0]
        v = v[1:2] + guidance * (v[0:1] - v[1:2])
        x0 = np.clip(math.sqrt(a) * x - math.sqrt(1 - a) * v, -6, 6)
        eps = math.sqrt(1 - a) * x + math.sqrt(a) * v
        x = (math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps).astype(np.float32)
    payload = {
        "generator": "frontend/scripts/fixtures/make_text2voxel_fixtures.py --sampler-only",
        "prior_sha256": hashlib.sha256((MODEL / "prior.onnx").read_bytes()).hexdigest(),
        "model_created": meta.get("created"),
        "prompt": ref["prompt"],
        "cond": b64_floats(cond),
        "latent": b64_floats(x[0]),
    }
    SAMPLER_OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {SAMPLER_OUT.relative_to(REPO)} for prior {payload['prior_sha256'][:12]}")


def main() -> None:
    if "--sampler-only" in sys.argv:
        write_sampler_reference()
        return
    from tokenizers import Tokenizer  # noqa: PLC0415

    tokenizer = Tokenizer.from_file(str(MINILM / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(MAX_TOKENS)

    captions = HAND_CAPTIONS + generated_captions(130)
    strings = (captions + TEST_PROMPTS + template_sample(30) + EDGE_CASES + UNICODE_CASES
               + random_strings(70))
    # De-duplicate but keep order.
    strings = list(dict.fromkeys(strings))
    token_cases = [{"text": s, "ids": tokenizer.encode(s).ids} for s in strings]

    # Embeddings: one string per call, exactly as the browser runs a prompt
    # (batching would pad, and padding shifts the int8 model's dynamic
    # quantisation scales).
    encoder = TextEncoder(MINILM)
    embed_strings = (["a red office chair with armrests and wheels"] + TEST_PROMPTS
                     + HAND_CAPTIONS[1:12] + ["", "café naïve", "中文椅子", "a chair 🪑",
                                              " ".join(["chair"] * 140), "[MASK] table"])
    embed_strings = list(dict.fromkeys(embed_strings))
    embeddings = [
        {"text": s, "embedding": b64_floats(encoder.encode([s], batch_size=1)[0])}
        for s in embed_strings
    ]

    timesteps = {
        str(steps): np.linspace(999, 0, steps).round().astype(int).tolist()
        for steps in (1, 2, 3, 7, 10, 20, 25, 49, 50, 75, 100, 250, 999, 1000)
    }

    fixtures = {
        "generator": "frontend/scripts/fixtures/make_text2voxel_fixtures.py",
        "tokenizers_version": __import__("tokenizers").__version__,
        "unicodedata_version": unicodedata.unidata_version,
        "max_tokens": MAX_TOKENS,
        "caption_count": len(captions),
        "tokens": token_cases,
        "embeddings": embeddings,
        "timesteps": timesteps,
        "dimensions": dimension_cases(),
        "sheets": sheet_cases(),
        "modelnet_names": sorted(MODELNET_NAMES),
        "palette": sorted(PALETTE),
    }
    FIXTURE_OUT.write_text(json.dumps(fixtures, ensure_ascii=True), encoding="utf-8")

    probe = probe_unicode(tokenizer)
    PROBE_OUT.write_text(json.dumps(probe, ensure_ascii=True), encoding="utf-8")
    write_tables(probe)

    print(f"{len(token_cases)} tokenizer strings ({len(captions)} captions), "
          f"{len(embeddings)} embeddings, {len(fixtures['dimensions'])} dimension cases, "
          f"{len(fixtures['sheets'])} sheets")
    print(f"probe: {len(probe['removed'])} removed ranges, {len(probe['chinese'])} CJK ranges, "
          f"{len(probe['marks'])} mark ranges, {len(probe['punct'])} punctuation ranges, "
          f"{len(probe['normalized_digests'])} digest blocks")
    for path in (FIXTURE_OUT, PROBE_OUT, TABLES_OUT):
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size:,} bytes)")
    if (MODEL / "prior.onnx").exists():
        write_sampler_reference()


if __name__ == "__main__":
    main()
