"""Prompt understanding: natural language -> :class:`DesignSpec`.

This is a deterministic, rule-based extraction layer.  It runs offline with no
model weights, which keeps the first pipeline stage fast and reproducible and
makes it straightforward to unit-test.

The original research notebook proposed BERT/RoBERTa for this stage.  A
transformer classifier can be dropped in behind the same :func:`parse_prompt`
signature; see ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import logging
import re

from app.errors import ValidationError
from app.schemas import DesignSpec, Dimensions, ViewName

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------

MATERIALS: dict[str, str] = {
    "stainless steel": "stainless steel",
    "carbon fibre": "carbon fibre",
    "carbon fiber": "carbon fibre",
    "aluminium": "aluminium",
    "aluminum": "aluminium",
    "polycarbonate": "polycarbonate",
    "polypropylene": "polypropylene",
    "titanium": "titanium",
    "magnesium": "magnesium",
    "plywood": "plywood",
    "bamboo": "bamboo",
    "walnut": "walnut",
    "silicone": "silicone",
    "ceramic": "ceramic",
    "concrete": "concrete",
    "leather": "leather",
    "textile": "fabric",
    "plastic": "plastic",
    "rubber": "rubber",
    "nylon": "nylon",
    "bronze": "bronze",
    "copper": "copper",
    "brass": "brass",
    "steel": "steel",
    "glass": "glass",
    "fabric": "fabric",
    "resin": "resin",
    "foam": "foam",
    "wood": "wood",
    "oak": "oak",
    "petg": "PETG",
    "abs": "ABS plastic",
    "pla": "PLA",
}

COLOURS: tuple[str, ...] = (
    "matte black", "gloss black", "matte white", "gloss white",
    "charcoal", "burgundy", "turquoise", "chrome", "bronze",
    "silver", "yellow", "orange", "purple", "beige", "cream",
    "ivory", "olive", "black", "white", "green", "brown", "navy",
    "teal", "gold", "grey", "gray", "blue", "pink", "red",
)

STYLES: tuple[str, ...] = (
    "minimalist", "minimal", "modern", "futuristic", "industrial",
    "ergonomic", "retro", "vintage", "organic", "brutalist",
    "scandinavian", "art deco", "sleek", "rugged", "utilitarian",
    "streamlined", "aerodynamic", "geometric", "biomimetic", "compact",
)

MANUFACTURING: dict[str, str] = {
    "3d print": "additive manufacturing (3D printing)",
    "3d-print": "additive manufacturing (3D printing)",
    "additive manufactur": "additive manufacturing (3D printing)",
    "injection mold": "injection moulding",
    "injection mould": "injection moulding",
    "sheet metal": "sheet metal fabrication",
    "vacuum form": "vacuum forming",
    "laser cut": "laser cutting",
    "die cast": "die casting",
    "machined": "CNC machining",
    "extruded": "extrusion",
    "welded": "welding",
    "cnc": "CNC machining",
    "fdm": "FDM 3D printing",
    "sla": "SLA resin printing",
    "sls": "SLS printing",
}

ACCESSIBILITY: dict[str, str] = {
    "wheelchair": "wheelchair compatible",
    "one-handed": "single-handed operation",
    "one handed": "single-handed operation",
    "single-handed": "single-handed operation",
    "one leg": "suitable for a single-leg user",
    "amputee": "suitable for limb-difference users",
    "prosthetic": "prosthetic compatible",
    "assistive": "assistive technology",
    "disabled": "designed for users with disabilities",
    "disability": "designed for users with disabilities",
    "mobility": "mobility support",
    "arthritis": "low grip-strength operation",
    "low vision": "low-vision accessible",
    "blind": "non-visual operation",
    "elderly": "suitable for older adults",
    "adaptive": "adaptive design",
    "easy grip": "easy-grip handling",
    "tremor": "tremor-tolerant controls",
}

CONSTRAINT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bwaterproof\b", "waterproof"),
    (r"\bwater[- ]resistant\b", "water resistant"),
    (r"\bweather[- ]?(proof|resistant)\b", "weather resistant"),
    (r"\bfoldable\b|\bfolds\b|\bfolding\b", "foldable"),
    (r"\bcollapsible\b", "collapsible"),
    (r"\bportable\b", "portable"),
    (r"\blightweight\b|\blight[- ]weight\b", "lightweight"),
    (r"\bdurable\b|\bheavy[- ]duty\b", "durable"),
    (r"\bstackable\b", "stackable"),
    (r"\bmodular\b", "modular"),
    (r"\badjustable\b", "adjustable"),
    (r"\btool[- ]free\b", "tool-free assembly"),
    (r"\bflame[- ]retardant\b", "flame retardant"),
    (r"\bfood[- ]safe\b", "food safe"),
    (r"\brecyclable\b", "recyclable"),
    (r"\bindoor\b", "indoor use"),
    (r"\boutdoor\b", "outdoor use"),
)

# Verbs that commonly open a design request and carry no object information.
LEAD_VERBS = (
    "design", "create", "generate", "make", "build", "model", "produce",
    "draw", "develop", "invent", "prototype", "sketch",
)

# Words that terminate the object noun phrase.
OBJECT_STOPWORDS = (
    " for ", " with ", " that ", " which ", " made ", " using ", " in ",
    " to ", " so ", " able ", " capable ", " featuring ", " having ",
)

# --------------------------------------------------------------------------
# Unit handling
# --------------------------------------------------------------------------

_TO_MM: dict[str, float] = {
    "mm": 1.0, "millimetre": 1.0, "millimeter": 1.0, "millimetres": 1.0, "millimeters": 1.0,
    "cm": 10.0, "centimetre": 10.0, "centimeter": 10.0, "centimetres": 10.0, "centimeters": 10.0,
    "m": 1000.0, "metre": 1000.0, "meter": 1000.0, "metres": 1000.0, "meters": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4,
    "ft": 304.8, "foot": 304.8, "feet": 304.8, "'": 304.8,
}

_TO_GRAMS: dict[str, float] = {
    "g": 1.0, "gram": 1.0, "grams": 1.0, "gramme": 1.0, "grammes": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0, "kilo": 1000.0, "kilos": 1000.0,
    "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
}

_UNIT_ALT = "|".join(sorted((u for u in _TO_MM if u.isalpha()), key=len, reverse=True))
_MASS_ALT = "|".join(sorted(_TO_GRAMS, key=len, reverse=True))


def _to_mm(value: float, unit: str) -> float | None:
    factor = _TO_MM.get(unit.lower().strip())
    return round(value * factor, 3) if factor else None


def _to_grams(value: float, unit: str) -> float | None:
    factor = _TO_GRAMS.get(unit.lower().strip())
    return round(value * factor, 3) if factor else None


# --------------------------------------------------------------------------
# Extraction helpers
# --------------------------------------------------------------------------


def _extract_triple(text: str) -> tuple[float, float, float] | None:
    """Match ``50 x 30 x 20 cm`` or ``50cm x 30cm x 20cm`` style dimensions."""
    triple = re.search(
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})?\s*[x×]\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})?\s*[x×]\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})",
        text,
        re.IGNORECASE,
    )
    if not triple:
        return None
    trailing_unit = triple.group(6)
    values: list[float] = []
    for number, unit in ((triple.group(1), triple.group(2)),
                         (triple.group(3), triple.group(4)),
                         (triple.group(5), triple.group(6))):
        millimetres = _to_mm(float(number), unit or trailing_unit)
        if millimetres is None:
            return None
        values.append(millimetres)
    return values[0], values[1], values[2]


_AXIS_WORDS: dict[str, tuple[str, ...]] = {
    "length_mm": ("long", "length", "deep", "depth"),
    "width_mm": ("wide", "width", "across"),
    "height_mm": ("tall", "high", "height"),
}


def _extract_named_axes(text: str) -> dict[str, float]:
    """Match ``1.2 m tall``, ``height of 40 cm``, ``40cm wide``."""
    found: dict[str, float] = {}
    for field, words in _AXIS_WORDS.items():
        word_alt = "|".join(words)
        # "<number><unit> tall"  /  "<number> <unit> in height"
        match = re.search(
            rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})\s*(?:in\s+)?(?:{word_alt})\b",
            text, re.IGNORECASE,
        )
        if not match:
            # "height of <number><unit>"  /  "a height of 40 cm"
            match = re.search(
                rf"(?:{word_alt})\s*(?:of|:|=)?\s*(\d+(?:\.\d+)?)\s*({_UNIT_ALT})\b",
                text, re.IGNORECASE,
            )
        if match:
            millimetres = _to_mm(float(match.group(1)), match.group(2))
            if millimetres is not None:
                found[field] = millimetres
    return found


#: Words that make a mass a *load rating* rather than the product's own weight.
_LOAD_CONTEXT = re.compile(
    r"\b(?:support|supports|supporting|load|loads|capacity|carry|carries|"
    r"carrying|hold|holds|holding|rated|bear|bears|up to a load of)\b",
    re.IGNORECASE,
)


def _extract_weight(text: str) -> float | None:
    """Extract the product's own mass in grams.

    A phrase such as "must support 300 kg" states a load capacity, not the
    item's weight, so masses introduced by a load verb are ignored - putting
    them in ``weight_g`` would corrupt the specification and every prompt
    built from it.
    """
    for match in re.finditer(
        rf"(?:under|below|less than|max(?:imum)?(?: of)?|weigh(?:s|ing|t)?(?: of)?|"
        rf"no more than|up to)?\s*(\d+(?:\.\d+)?)\s*({_MASS_ALT})\b",
        text, re.IGNORECASE,
    ):
        # Look back only as far as the current clause, so a load verb in an
        # earlier clause cannot suppress a later, genuine weight.
        preceding = re.split(r"[,;.]", text[:match.start()])[-1][-30:]
        if _LOAD_CONTEXT.search(preceding):
            continue
        return _to_grams(float(match.group(1)), match.group(2))
    return None


def _extract_dimensions(text: str) -> Dimensions:
    dimensions = Dimensions()
    triple = _extract_triple(text)
    if triple:
        dimensions.length_mm, dimensions.width_mm, dimensions.height_mm = triple
    for field, value in _extract_named_axes(text).items():
        setattr(dimensions, field, value)
    weight = _extract_weight(text)
    if weight is not None:
        dimensions.weight_g = weight
    return dimensions


#: Inflections a process verb may carry: print -> printable / printing / printed.
_SUFFIXES = r"(?:e?[sd]|ing|able|ible|ion|ure)?"


def _extract_vocabulary(text: str, vocabulary: dict[str, str],
                        *, inflected: bool = False) -> list[str]:
    """Return canonical labels for every vocabulary key present in ``text``.

    Longer keys are matched first so ``stainless steel`` wins over ``steel``.

    Args:
        inflected: allow a trailing inflection on the key, so "3D printable"
            and "casting" match. Left off for materials, where it would let
            "PLA" match "plastic".
    """
    results: list[str] = []
    consumed = text
    for key in sorted(vocabulary, key=len, reverse=True):
        pattern = re.escape(key)
        # Only require word boundaries where the key actually starts/ends with
        # a word character, so entries such as "3d-print" still match.
        left = r"\b" if key[0].isalnum() else ""
        right = ""
        if key[-1].isalnum():
            right = (_SUFFIXES + r"\b") if inflected else r"\b"
        if re.search(f"{left}{pattern}{right}", consumed, re.IGNORECASE):
            label = vocabulary[key]
            if label not in results:
                results.append(label)
            consumed = re.sub(f"{left}{pattern}{right}", " ", consumed, flags=re.IGNORECASE)
    return results


def _extract_first(text: str, candidates: tuple[str, ...]) -> str:
    for candidate in sorted(candidates, key=len, reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", text, re.IGNORECASE):
            return candidate
    return ""


def _extract_constraints(text: str) -> list[str]:
    constraints: list[str] = []
    for pattern, label in CONSTRAINT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and label not in constraints:
            constraints.append(label)

    # Explicit "must be ..." / "should be ..." clauses.
    for match in re.finditer(r"\b(?:must|should|needs? to)\s+(?:be\s+)?([^.,;]{3,60})",
                             text, re.IGNORECASE):
        clause = match.group(1).strip().rstrip(".")
        if clause and clause.lower() not in {c.lower() for c in constraints}:
            constraints.append(clause)
    return constraints[:20]


def _extract_object(text: str) -> str:
    """Pull the head noun phrase naming the product."""
    working = text.strip().rstrip(".!?")

    # Drop a leading imperative verb, then any article. Alternation is
    # first-match-wins, so the longer article must come first or "a" would
    # swallow only the "a" of "an" and leave a stray "n".
    verb_alt = "|".join(LEAD_VERBS)
    working = re.sub(
        rf"^\s*(?:please\s+)?(?:can you\s+|could you\s+|i (?:want|need)\s+)?"
        rf"(?:{verb_alt})(?:\s+me)?\s+",
        "", working, flags=re.IGNORECASE,
    )
    working = re.sub(r"^\s*(?:some|the|an|a)\s+", "", working, flags=re.IGNORECASE)

    # Cut at the first structural stopword, or the first clause break.
    lowered = f" {working.lower()} "
    cut = len(working)
    for stopword in OBJECT_STOPWORDS:
        index = lowered.find(stopword)
        if index != -1:
            cut = min(cut, max(index - 1, 0))
    comma = working.find(",")
    if comma > 0:
        cut = min(cut, comma)
    phrase = working[:cut].strip(" ,;:-")

    # Strip leading descriptive adjectives that already have dedicated fields,
    # so "futuristic matte black car" leaves just "car". Repeat until stable:
    # a single pass per vocabulary would miss a colour sitting behind a style.
    descriptors = sorted((*COLOURS, *STYLES), key=len, reverse=True)
    changed = True
    while changed and phrase:
        changed = False
        for word in descriptors:
            stripped = re.sub(rf"^\s*{re.escape(word)}\s+", "", phrase, flags=re.IGNORECASE)
            if stripped != phrase:
                phrase = stripped
                changed = True
                break

    phrase = re.sub(r"\s+", " ", phrase).strip(" ,;:-")
    if not phrase:
        phrase = re.sub(r"\s+", " ", working).strip()[:80] or "product"
    return phrase[:200]


def _extract_purpose(text: str) -> str:
    """Text following ``for`` / ``used for`` / ``so that`` describes intent."""
    # Stop at the clause break: a trailing comma usually starts a new fact
    # ("for a small apartment, 45 x 45 cm") that is not part of the purpose.
    match = re.search(
        r"\b(?:intended for|designed for|used for|for use in|for)\s+([^.;,]{3,300})",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".,;")[:1000]
    match = re.search(r"\b(?:so that|in order to|to)\s+([^.;,]{5,300})", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".,;")[:1000]
    return ""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def parse_prompt(prompt: str) -> DesignSpec:
    """Convert a free-text design request into a structured specification.

    Raises:
        ValidationError: if the prompt is empty or too short to interpret.
    """
    if prompt is None or not prompt.strip():
        raise ValidationError(
            "Enter a description of the product you want to design.",
            hint="For example: 'A lightweight ergonomic wheelchair tray in matte black ABS'.",
        )
    text = re.sub(r"\s+", " ", prompt.strip())
    if len(text) < 3:
        raise ValidationError(
            "That prompt is too short to interpret.",
            hint="Describe the object, what it is for, and any size or material requirements.",
        )

    tolerance = None
    tolerance_match = re.search(
        rf"(?:tolerance|accuracy|precision)\s*(?:of|:|=|±)?\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_ALT})",
        text, re.IGNORECASE,
    )
    if tolerance_match:
        tolerance = _to_mm(float(tolerance_match.group(1)), tolerance_match.group(2))
        if tolerance is not None and not 0 < tolerance <= 100:
            tolerance = None

    spec = DesignSpec(
        object=_extract_object(text),
        purpose=_extract_purpose(text),
        dimensions=_extract_dimensions(text),
        materials=_extract_vocabulary(text, MATERIALS),
        style=_extract_first(text, STYLES),
        color=_extract_first(text, COLOURS),
        constraints=_extract_constraints(text),
        manufacturing_requirements=_extract_vocabulary(text, MANUFACTURING, inflected=True),
        accessibility_requirements=_extract_vocabulary(text, ACCESSIBILITY),
        intended_use=_extract_purpose(text)[:500],
        tolerances_mm=tolerance,
    )
    return _apply_semantics(spec, text)


def _apply_semantics(spec: DesignSpec, text: str) -> DesignSpec:
    """Refine a rule-parsed spec with the transformer layer.

    Rules stay authoritative for anything they matched confidently; the
    semantic pass only repairs the object phrase and fills vocabularies the
    rules left empty. If the encoder is unavailable this is a no-op, so the
    parser keeps working offline.
    """
    from app.config import get_settings  # noqa: PLC0415

    if not get_settings().semantic_prompt:
        return spec

    from app.pipeline import semantic  # noqa: PLC0415

    canonical, category, score = semantic.canonicalise_object(spec.object)
    if category and canonical and canonical.lower() != spec.object.lower():
        logger.info("Prompt restructured: %r -> %r", spec.object, canonical)
        spec.object = canonical[:200]
        # The corrected wording often reveals vocabulary the raw text hid -
        # "wheelchair" only appears once the typo is repaired.
        combined = f"{text} {canonical}"
        for field, vocabulary, inflected in (
            ("accessibility_requirements", ACCESSIBILITY, False),
            ("manufacturing_requirements", MANUFACTURING, True),
            ("materials", MATERIALS, False),
        ):
            merged = list(dict.fromkeys(
                [*getattr(spec, field), *_extract_vocabulary(combined, vocabulary,
                                                             inflected=inflected)]
            ))
            setattr(spec, field, merged[:20])

    if not spec.materials:
        spec.materials = semantic.enrich_vocabulary(text, sorted(set(MATERIALS.values())))[:4]
    if not spec.accessibility_requirements:
        spec.accessibility_requirements = semantic.enrich_vocabulary(
            text, sorted(set(ACCESSIBILITY.values()))
        )[:4]
    return spec


# --------------------------------------------------------------------------
# Prompt construction for multi-view image generation
# --------------------------------------------------------------------------

VIEW_PHRASES: dict[ViewName, str] = {
    ViewName.FRONT: "flat front elevation seen exactly head-on at 90 degrees",
    ViewName.REAR: "flat rear elevation seen exactly from behind at 90 degrees",
    ViewName.LEFT: "flat left side profile seen from exactly 90 degrees to the side",
    ViewName.RIGHT: "flat right side profile seen from exactly 90 degrees to the side",
    ViewName.TOP: "flat top-down plan view seen from exactly overhead",
    ViewName.BOTTOM: "flat bottom-up plan view seen from exactly underneath",
    ViewName.PERSPECTIVE: "three-quarter perspective hero view, isometric product shot",
}

#: Silhouette carving assumes each view is an orthographic projection along its
#: own axis. Left to itself a diffusion model returns a three-quarter marketing
#: shot for almost any vehicle or product, and a three-quarter "front" shows the
#: object's full length - which the carver then reads as width, collapsing the
#: model into a squat blob.
#:
#: Turbo checkpoints run at guidance 0, where a negative prompt has no effect,
#: so the negations have to live in the positive prompt to do any work.
ORTHOGRAPHIC_CLAUSE = (
    "no perspective distortion, no three-quarter angle, orthographic projection, "
    "camera perpendicular to the object"
)

# Shared across every view so the object keeps one identity between renders.
STYLE_ANCHOR = (
    "industrial design product render, single isolated object, "
    "plain flat white seamless background, even studio lighting, "
    "no shadows on background, no text, no watermark, no people, "
    "full object visible, sharp focus, high detail"
)

NEGATIVE_PROMPT = (
    "multiple objects, collage, grid, split screen, cropped, cut off, "
    "text, watermark, signature, logo, people, hands, cluttered background, "
    "busy background, gradient background, shadow, reflection, blurry, "
    "low quality, distorted, deformed"
)


def spec_to_base_prompt(spec: DesignSpec) -> str:
    """Build the shared identity clause reused by every view."""
    parts: list[str] = []
    descriptor = " ".join(word for word in (spec.color, spec.style, spec.object) if word)
    parts.append(descriptor.strip() or spec.object)

    if spec.materials:
        parts.append("made of " + ", ".join(spec.materials))
    dimension_phrase = spec.dimensions.described()
    if dimension_phrase:
        parts.append(dimension_phrase)
    if spec.purpose:
        parts.append(f"for {spec.purpose}")
    if spec.accessibility_requirements:
        parts.append(", ".join(spec.accessibility_requirements))
    if spec.constraints:
        parts.append(", ".join(spec.constraints[:4]))
    return ", ".join(part for part in parts if part)


def build_view_prompt(spec: DesignSpec, view: ViewName) -> str:
    """Compose the full positive prompt for one camera view.

    The camera phrase leads. Diffusion models weight early tokens far more
    heavily, and with a few-step turbo checkpoint a trailing "front view" is
    largely ignored - it returns a three-quarter product shot instead of the
    elevation the reconstruction needs. Putting the view first measurably
    improves adherence, which is what the silhouette carving depends on.
    """
    clause = "" if view is ViewName.PERSPECTIVE else f"{ORTHOGRAPHIC_CLAUSE}, "
    return f"{VIEW_PHRASES[view]} of a {spec_to_base_prompt(spec)}, {clause}{STYLE_ANCHOR}"
