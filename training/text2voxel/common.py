"""Shared definitions for the Text2Voxel training kernels.

Kaggle script kernels are single files, so ``training/text2voxel/build_kernels.py`` inlines
this module into each kernel's generated ``kernel.py``. Anything the browser or
the backend must reproduce exactly (text embedding, caption templates, colour
palette) is defined here once.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Text encoder: the exact file the browser and the backend load.
# --------------------------------------------------------------------------

#: all-MiniLM-L6-v2, int8-quantised ONNX export. Pinned by commit so training
#: embeddings are bit-for-bit reproducible at inference.
MINILM_REPO = "Xenova/all-MiniLM-L6-v2"
MINILM_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
MINILM_FILES = ("onnx/model_quantized.onnx", "tokenizer.json")
MAX_TOKENS = 128
EMBED_DIM = 384


def ensure_packages(*names: str) -> None:
    """pip-install missing modules (Kaggle's image lacks onnxruntime/onnx)."""
    import importlib.util
    import subprocess
    import sys

    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if missing:
        print("pip install", *missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def fetch_minilm(target: Path) -> Path:
    """Download the pinned MiniLM files into ``target`` (idempotent)."""
    target.mkdir(parents=True, exist_ok=True)
    for name in MINILM_FILES:
        out = target / Path(name).name
        if out.exists() and out.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{MINILM_REPO}/resolve/{MINILM_REVISION}/{name}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, out)
    return target


class TextEncoder:
    """Mean-pooled, L2-normalised MiniLM sentence embeddings via onnxruntime."""

    def __init__(self, model_dir: Path, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(MAX_TOKENS)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / "model_quantized.onnx"), options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encodings = self.tokenizer.encode_batch(chunk)
            width = max(len(e.ids) for e in encodings)
            ids = np.zeros((len(chunk), width), dtype=np.int64)
            mask = np.zeros((len(chunk), width), dtype=np.int64)
            for row, encoding in enumerate(encodings):
                ids[row, :len(encoding.ids)] = encoding.ids
                mask[row, :len(encoding.ids)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out[start:start + len(chunk)] = pooled
        return out


# --------------------------------------------------------------------------
# Caption templates for the uncaptioned ModelNet40 shapes.
# --------------------------------------------------------------------------

#: Readable names per ModelNet40 class. The first entry is the canonical label.
MODELNET_NAMES: dict[str, list[str]] = {
    "airplane": ["airplane", "aeroplane", "aircraft", "jet plane"],
    "bathtub": ["bathtub", "bath tub"],
    "bed": ["bed", "double bed"],
    "bench": ["bench", "park bench"],
    "bookshelf": ["bookshelf", "bookcase", "shelving unit"],
    "bottle": ["bottle"],
    "bowl": ["bowl"],
    "car": ["car", "automobile", "sports car"],
    "chair": ["chair"],
    "cone": ["cone", "traffic cone"],
    "cup": ["cup", "mug"],
    "curtain": ["curtain"],
    "desk": ["desk", "writing desk"],
    "door": ["door"],
    "dresser": ["dresser", "chest of drawers"],
    "flower_pot": ["flower pot", "planter"],
    "glass_box": ["glass box", "display case"],
    "guitar": ["guitar"],
    "keyboard": ["keyboard", "computer keyboard"],
    "lamp": ["lamp", "floor lamp", "desk lamp"],
    "laptop": ["laptop", "laptop computer"],
    "mantel": ["mantel", "fireplace mantel"],
    "monitor": ["monitor", "computer monitor", "display screen"],
    "night_stand": ["nightstand", "bedside table"],
    "person": ["person", "human figure"],
    "piano": ["piano", "grand piano"],
    "plant": ["plant", "potted plant"],
    "radio": ["radio"],
    "range_hood": ["range hood", "cooker hood"],
    "sink": ["sink", "wash basin"],
    "sofa": ["sofa", "couch"],
    "stairs": ["stairs", "staircase"],
    "stool": ["stool", "bar stool"],
    "table": ["table"],
    "tent": ["tent"],
    "toilet": ["toilet"],
    "tv_stand": ["tv stand", "media console"],
    "vase": ["vase"],
    "wardrobe": ["wardrobe", "armoire"],
    "xbox": ["xbox", "game console"],
}

#: Named colours for synthetic uniform-colour augmentation of ModelNet shapes
#: (which have no texture). RGB in 0-255.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (200, 40, 40),
    "orange": (230, 120, 30),
    "yellow": (230, 200, 40),
    "green": (50, 150, 60),
    "blue": (40, 80, 190),
    "purple": (120, 60, 160),
    "pink": (230, 130, 170),
    "brown": (120, 75, 40),
    "black": (30, 30, 30),
    "white": (235, 235, 235),
    "grey": (130, 130, 130),
    "silver": (190, 192, 198),
}
#: Colour used when a caption names no colour (untextured CAD look).
NEUTRAL_RGB = (175, 175, 180)

PLAIN_TEMPLATES = ("{n}", "{a} {n}", "{a} 3d model of {a2} {n}", "the {n}")
COLOUR_TEMPLATES = ("{a} {c} {n}", "{a} {n} in {c}", "{c} {n}")


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def plain_captions(name: str) -> list[str]:
    return [t.format(n=name, a=article(name), a2=article(name)) for t in PLAIN_TEMPLATES]


def colour_captions(name: str, colour: str) -> list[str]:
    out = []
    for template in COLOUR_TEMPLATES:
        # The article agrees with the word that follows it.
        first = colour if template.startswith("{a} {c}") else name
        out.append(template.format(n=name, c=colour, a=article(first)))
    return out


def all_template_strings() -> list[str]:
    strings: set[str] = set()
    for names in MODELNET_NAMES.values():
        for name in names:
            strings.update(plain_captions(name))
            for colour in PALETTE:
                strings.update(colour_captions(name, colour))
    for name in ("chair", "table"):
        strings.update(plain_captions(name))
    return sorted(strings)


# --------------------------------------------------------------------------
# Small I/O helpers
# --------------------------------------------------------------------------


def find_input(pattern: str, root: str | None = None) -> Path:
    """Locate one file under the Kaggle input mounts (layout varies)."""
    root = root or os.environ.get("T2V_INPUT", "/kaggle/input")
    matches = sorted(Path(root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return matches[0]


def _finite(value):
    """NaN/inf -> None, recursively: browsers' JSON.parse rejects bare NaN."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_finite(payload), indent=2, default=str, allow_nan=False))


def cpu_count() -> int:
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
