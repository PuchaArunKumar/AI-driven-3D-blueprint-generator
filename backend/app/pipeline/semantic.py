"""Transformer-backed prompt understanding.

The rule-based parser in :mod:`app.pipeline.prompt_parser` is fast, exact and
offline, but it only recognises wording it was written for. It misses
synonyms ("beech" is wood), paraphrases ("for someone who cannot grip well"
is a low grip-strength requirement) and typos.

That last one is not cosmetic. A prompt of *"a electric wheel cchair"* leaves
the object as the literal string ``electric wheel cchair``, which goes
straight into the image prompt - and a diffusion model reading it latches onto
"wheel" and draws a car wheel. The whole pipeline then faithfully reconstructs
a wheel.

This module adds a semantic layer over the rules, using a BERT-family sentence
encoder (MiniLM: a 6-layer distilled BERT) to:

* **canonicalise the object** against a product-category vocabulary, so
  "electric wheel cchair" is restructured to "electric wheelchair";
* **enrich the vocabularies** by meaning rather than spelling.

It augments and never overrides a confident rule match, and it is entirely
optional: if torch/transformers or the weights are unavailable, everything
falls back to the rules. Embeddings are matched alongside a character-level
ratio, because embeddings alone handle synonyms well but typos poorly.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

#: A distilled BERT. Small (~88 MB), CPU-fast, and good at short-phrase
#: similarity, which is all this needs.
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

#: Weight given to the sentence embedding when scoring a category. The
#: remainder goes to a character-level ratio: embeddings capture meaning
#: ("wheel chair" -> wheelchair) but subword tokenisation handles typos badly,
#: while a character ratio handles typos and little else. Together they cover
#: both, and each alone was measured to fail on the other's cases.
EMBEDDING_WEIGHT = 0.55

#: Blended score above which a category is accepted. Measured: genuine repairs
#: ("electric wheel cchair", "wheelchiar") score 0.69-0.71, while forcing an
#: unlisted object onto its nearest neighbour ("cup" -> mug) scores 0.50-0.54.
#: Leaving a novel object alone is always better than mislabelling it.
OBJECT_THRESHOLD = 0.62
#: Character-ratio above which a token is considered part of the category
#: already ("wheel" and "cchair" are both fragments of "wheelchair").
ABSORB_RATIO = 0.62
#: Similarity required to add a vocabulary term the rules did not find.
#: Measured over paraphrase prompts ("cannot grip well", "weak hands") against
#: unrelated ones (brackets, cars, lamps): 0.50 recalls 4/4 with 0/4 false
#: positives, while 0.60 recalls only 2/4. Short abstract labels sit low in
#: embedding space, so the usual 0.6+ rule of thumb is too strict here.
VOCABULARY_THRESHOLD = 0.50

#: Product categories the studio is meant to design. Canonicalising against
#: this list is what turns a typo'd noun phrase into something a diffusion
#: model can actually draw.
PRODUCT_CATEGORIES: tuple[str, ...] = (
    # accessibility and assistive technology
    "wheelchair", "electric wheelchair", "wheelchair ramp", "wheelchair tray",
    "mobility scooter", "walking frame", "walking stick", "crutch",
    "prosthetic limb", "orthopaedic brace", "hearing aid", "shower seat",
    "grab rail", "adaptive cutlery", "pill dispenser", "reacher grabber",
    # furniture
    "chair", "dining chair", "office chair", "stool", "bench", "table",
    "desk", "coffee table", "shelf", "bookcase", "cabinet", "bed frame",
    "wardrobe", "lamp", "floor lamp",
    # vehicles
    "car", "sports car", "electric car", "van", "truck", "bus",
    "bicycle", "bicycle frame", "motorcycle", "scooter", "skateboard",
    "wheel", "tyre", "helmet",
    # consumer and wearable
    "smartwatch", "fitness band", "headphones", "earbuds", "speaker",
    "phone stand", "laptop stand", "backpack", "water bottle", "mug",
    "vase", "clock", "camera", "keyboard", "mouse",
    # mechanical and industrial
    "bracket", "enclosure", "housing", "gearbox", "robot arm", "drone",
    "propeller", "gear", "flange", "pipe fitting", "hinge", "clamp",
    "tool handle", "3d printer part",
)

_lock = threading.Lock()


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _encoder():
    """Load the sentence encoder once, or return None when unavailable."""
    try:
        import torch  # noqa: F401,PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    except ImportError:
        logger.info("transformers/torch unavailable; prompt understanding stays rule-based")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).eval()
    except Exception:
        logger.info(
            "Could not load %s; prompt understanding stays rule-based. "
            "It downloads automatically on first use when online.", MODEL_ID,
        )
        return None

    logger.info("Semantic prompt understanding enabled (%s)", MODEL_ID)
    return tokenizer, model


def available() -> bool:
    """Whether the semantic layer can run right now."""
    return _encoder() is not None


def embed(texts: list[str]) -> np.ndarray | None:
    """Mean-pooled, L2-normalised sentence embeddings, or None if unavailable."""
    loaded = _encoder()
    if loaded is None or not texts:
        return None
    tokenizer, model = loaded

    import torch  # noqa: PLC0415

    with _lock, torch.no_grad():
        batch = tokenizer(texts, padding=True, truncation=True,
                          max_length=64, return_tensors="pt")
        output = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy()


@lru_cache(maxsize=1)
def _category_embeddings() -> np.ndarray | None:
    return embed(list(PRODUCT_CATEGORIES))


# --------------------------------------------------------------------------
# Object canonicalisation
# --------------------------------------------------------------------------


def _absorbed(token: str, category: str) -> bool:
    """Is ``token`` already accounted for by ``category``?

    "wheel" and "cchair" are both fragments of "wheelchair", so neither should
    survive as a separate descriptor once the category is known.
    """
    token = token.lower()
    if token in category:
        return True
    if difflib.SequenceMatcher(None, token, category).ratio() >= ABSORB_RATIO:
        return True
    return any(
        difflib.SequenceMatcher(None, token, part).ratio() >= ABSORB_RATIO
        for part in category.split()
    )


def _character_ratio(phrase: str, category: str) -> float:
    """Spelling closeness, ignoring spacing.

    Comparing with whitespace removed is what lets "wheel chair" score against
    "wheelchair" as highly as a straight typo would.
    """
    squash = lambda text: re.sub(r"[^a-z]", "", text.lower())  # noqa: E731
    return max(
        difflib.SequenceMatcher(None, phrase.lower(), category.lower()).ratio(),
        difflib.SequenceMatcher(None, squash(phrase), squash(category)).ratio(),
    )


def canonicalise_object(phrase: str) -> tuple[str, str | None, float]:
    """Rewrite an object phrase around its recognised product category.

    Returns ``(phrase, category, score)``. The phrase is unchanged when no
    category is confident enough, so a novel object is never mangled into the
    nearest thing on the list.
    """
    cleaned = re.sub(r"\s+", " ", (phrase or "").strip())
    if not cleaned:
        return phrase, None, 0.0

    lowered = cleaned.lower()
    # A whole-word mention needs no rewriting. Word boundaries matter: plain
    # substring matching finds "chair" inside "cchair" and "wheel" inside
    # "wheelchiar", which is exactly the mistake being corrected here.
    exact = [
        category for category in PRODUCT_CATEGORIES
        if re.search(r"\b" + re.escape(category) + r"\b", lowered)
    ]
    if exact:
        longest = max(exact, key=len)
        # Only trust it when it explains most of the phrase; "wheel" inside
        # "electric wheel cchair" explains very little.
        if len(longest) >= 0.5 * len(lowered.replace(" ", "")):
            return cleaned, longest, 1.0

    ratios = np.array([_character_ratio(lowered, c) for c in PRODUCT_CATEGORIES])
    categories = _category_embeddings()
    query = embed([lowered])

    if categories is None or query is None:
        # No encoder: fuzzy spelling alone still repairs typos.
        scores = ratios
    else:
        cosine = categories @ query[0]
        scores = EMBEDDING_WEIGHT * cosine + (1.0 - EMBEDDING_WEIGHT) * ratios

    best = int(np.argmax(scores))
    score = float(scores[best])
    if score < OBJECT_THRESHOLD:
        return cleaned, None, score

    category = PRODUCT_CATEGORIES[best]

    # If the phrase already uses the category's head noun, it is unambiguous
    # and needs no repair: "bike frame" must not become "bike bicycle frame".
    head = category.split()[-1]
    if re.search(r"\b" + re.escape(head) + r"\b", lowered):
        return cleaned, category, score
    # Substitute the category in place of the words it already covers, keeping
    # the original word order so descriptive context survives:
    # "robot arm for a factory" must not become "for factory robot arm".
    tokens = lowered.split()
    absorbed = [index for index, token in enumerate(tokens) if _absorbed(token, category)]
    if absorbed:
        rebuilt_tokens = [
            category if index == absorbed[0] else token
            for index, token in enumerate(tokens)
            if index == absorbed[0] or index not in absorbed
        ]
    else:
        rebuilt_tokens = [*tokens, category]
    rebuilt = " ".join(rebuilt_tokens).strip()
    logger.info("Canonicalised object %r -> %r (%.2f)", cleaned, rebuilt, score)
    return rebuilt, category, score


# --------------------------------------------------------------------------
# Vocabulary enrichment
# --------------------------------------------------------------------------


def _phrases(text: str) -> list[str]:
    """Split a prompt into clause-sized chunks to match against."""
    parts = re.split(r"[,;.]|\band\b|\bwith\b|\bfor\b", text.lower())
    return [part.strip() for part in parts if len(part.strip()) >= 3][:16]


def enrich_vocabulary(text: str, labels: list[str],
                      threshold: float = VOCABULARY_THRESHOLD) -> list[str]:
    """Return canonical labels whose *meaning* appears in ``text``.

    Complements exact matching: it is what recognises "beech" as wood, or
    "cannot grip well" as a low grip-strength requirement.
    """
    if not labels:
        return []
    phrases = _phrases(text)
    phrase_vectors = embed(phrases)
    label_vectors = embed(labels)
    if phrase_vectors is None or label_vectors is None:
        return []

    similarity = label_vectors @ phrase_vectors.T          # (labels, phrases)
    best = similarity.max(axis=1)
    return [label for label, score in zip(labels, best, strict=True) if score >= threshold]
