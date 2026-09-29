"""Text-to-3D with Text2Voxel-64, run locally on the CPU with onnxruntime.

Every other 3D provider reconstructs a model from view images. This one reads
the prompt itself, so the pipeline skips the image stage when it runs:

    prompt -> MiniLM sentence embedding (384-d, mean-pooled, L2-normalised)
           -> DDIM sampling of a latent diffusion prior (256-d, v-prediction,
              classifier-free guidance against an all-zero embedding)
           -> VAE decoder -> 64^3 occupancy probabilities + RGB
           -> floaters dropped, marching cubes at ``meta.threshold`` (lower
              only for a near-empty grid - see ``prepare_field``) -> coloured GLB

The sampler reproduces ``ddim_sample`` in ``training/text2voxel/k2_train.py``
step for step; ``tests/test_text2voxel.py`` checks it against the reference
latent stored in ``meta.json``. The browser engine runs the same files.

What to expect from it: the training data is Text2Shape (ShapeNet chairs and
tables with human captions) plus ModelNet40 (40 CAD categories with template
captions), so chairs and tables come out with real detail, the other 39
categories are coarse, and anything outside those categories gets the model's
nearest guess. 64^3 voxels is a concept-level shape, not a manufacturing
drawing. Physical size comes from the specification, never from the model.

The model files are not bundled and this provider never downloads anything.
Install them with::

    python scripts/install_text2voxel.py
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.errors import GenerationError, ProviderUnavailableError
from app.providers.base import (
    Availability,
    ModelResult,
    ProgressCallback,
    TextTo3DGenerator,
    _noop_progress,
)
from app.schemas import DesignSpec

logger = logging.getLogger(__name__)

INSTALL_HINT = "Run: python scripts/install_text2voxel.py (from the backend directory)"

#: Every file the provider needs, relative to the model directory.
MODEL_FILES = (
    "decoder.onnx",
    "prior.onnx",
    "meta.json",
    "minilm/model_quantized.onnx",
    "minilm/tokenizer.json",
)

#: Python packages the provider imports, as (module, pip name).
REQUIRED_PACKAGES = (
    ("onnxruntime", "onnxruntime"),
    ("tokenizers", "tokenizers"),
    ("skimage", "scikit-image"),
)

#: Progress budget: sampling dominates the runtime, so it gets most of the bar.
SAMPLING_START = 0.12
SAMPLING_END = 0.80

#: A prior callable: x [B, latent], t [B], cond [B, cond_dim] -> v [B, latent].
PriorFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

#: Surface preparation, identical to the browser engine's
#: (frontend/src/lib/text2voxel/mesher.ts ``prepareField``): connected blobs
#: under MIN_COMPONENT_VOXELS (26-connectivity) are dropped as floaters, the
#: largest always kept; when fewer than MIN_OCCUPIED_VOXELS reach the model's
#: threshold (the untrained placeholder does this) the surface is taken at the
#: lower level that encloses FALLBACK_VOXELS voxels, with a warning.
MIN_COMPONENT_VOXELS = 8
MIN_OCCUPIED_VOXELS = 64
FALLBACK_VOXELS = 4096


# --------------------------------------------------------------------------
# Maths shared with the training kernel and the browser engine
# --------------------------------------------------------------------------


def ddim_timesteps(train_steps: int, steps: int) -> np.ndarray:
    """The integer timesteps DDIM visits, from noisiest to clean.

    ``numpy.round`` rounds half to even, exactly as the training kernel does.
    """
    if steps < 1 or steps > train_steps:
        raise ValueError(f"steps must be between 1 and {train_steps}, got {steps}")
    return np.linspace(train_steps - 1, 0, steps).round().astype(int)


def ddim_sample(
    prior: PriorFn,
    cond: np.ndarray,
    noise: np.ndarray,
    alphas_cumprod: np.ndarray | list[float],
    *,
    steps: int,
    guidance: float,
    x0_clip: float = 6.0,
    on_step: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Deterministic DDIM with classifier-free guidance.

    Mirrors ``ddim_sample`` and the reference block in ``k2_train.main``. The
    conditional and unconditional branches run as one batch of two.
    ``on_step(index, total)`` fires after each step - the provider uses it for
    progress, which is also where a cancelled job stops.
    """
    alphas = np.asarray(alphas_cumprod, dtype=np.float64)
    times = ddim_timesteps(len(alphas), steps)
    x = np.asarray(noise, dtype=np.float32).reshape(-1)
    cond = np.asarray(cond, dtype=np.float32).reshape(-1)
    conds = np.stack([cond, np.zeros_like(cond)])

    for index, t in enumerate(times):
        a = float(alphas[t])
        a_prev = float(alphas[times[index + 1]]) if index + 1 < len(times) else 1.0
        v = np.asarray(
            prior(np.stack([x, x]), np.full(2, float(t), dtype=np.float32), conds),
            dtype=np.float32,
        )
        v_cond, v_uncond = v[0], v[1]
        v = v_uncond + guidance * (v_cond - v_uncond)
        x0 = np.clip(math.sqrt(a) * x - math.sqrt(1.0 - a) * v, -x0_clip, x0_clip)
        eps = math.sqrt(1.0 - a) * x + math.sqrt(a) * v
        x = (math.sqrt(a_prev) * x0 + math.sqrt(1.0 - a_prev) * eps).astype(np.float32)
        if on_step is not None:
            on_step(index, len(times))
    return x


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Attention-masked mean over tokens, then L2-normalise (``TextEncoder.encode``)."""
    weights = mask[..., None].astype(np.float32)
    pooled = (hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)
    pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    return pooled.astype(np.float32)


@dataclass
class ShapeField:
    """The occupancy grid the surface is extracted from, and how it was chosen."""

    field: np.ndarray       # [x][y][z] occupancy with dropped floaters zeroed
    iso: float              # the level the surface is extracted at
    occupied_voxels: int    # voxels at or above ``iso`` that were kept
    warnings: list[str]


def _empty_shape_error() -> GenerationError:
    return GenerationError(
        "Text2Voxel-64 produced an empty shape for this prompt.",
        hint="Try another seed, or describe the object more plainly - "
             "for example 'a wooden chair'.",
    )


def prepare_field(occupancy: np.ndarray, threshold: float = 0.5) -> ShapeField:
    """Choose the iso-level and drop floating fragments, as the browser does.

    See MIN_COMPONENT_VOXELS / MIN_OCCUPIED_VOXELS / FALLBACK_VOXELS.

    Raises:
        GenerationError: when the grid holds no occupancy at all.
    """
    from scipy import ndimage  # noqa: PLC0415

    occupancy = np.asarray(occupancy, dtype=np.float32)
    resolution = occupancy.shape[-1]
    grid = occupancy.reshape(resolution, resolution, resolution)
    warnings: list[str] = []

    iso = float(threshold)
    reached = int((grid >= iso).sum())
    if reached < MIN_OCCUPIED_VOXELS:
        ordered = np.sort(grid, axis=None)
        fallback = float(ordered[max(0, ordered.size - FALLBACK_VOXELS)])
        if not fallback > 0.0:
            raise _empty_shape_error()
        iso = min(fallback, iso)
        warnings.append(
            f"Only {reached} voxel{'' if reached == 1 else 's'} reached the model's "
            f"{threshold:g} occupancy threshold (peak {float(grid.max()):.3f}), so the "
            f"surface was extracted at {iso:.3f} instead. This happens with an untrained "
            "or placeholder model; the shape is not meaningful."
        )

    labels, count = ndimage.label(grid >= iso, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        raise _empty_shape_error()
    sizes = np.bincount(labels.ravel())[1:]
    keep = sizes >= MIN_COMPONENT_VOXELS
    keep[int(np.argmax(sizes))] = True
    field = grid.copy()
    dropped = np.flatnonzero(~keep)
    if dropped.size:
        field[np.isin(labels, dropped + 1)] = 0.0
        warnings.append(
            f"Dropped {dropped.size} floating fragment(s) ({int(sizes[dropped].sum())} voxels) "
            "smaller than the minimum part size."
        )
    return ShapeField(field=field, iso=iso, occupied_voxels=int(sizes[keep].sum()),
                      warnings=warnings)


def voxels_to_mesh(occupancy: np.ndarray, rgb: np.ndarray | None, threshold: float = 0.5):
    """Surface the occupancy grid and colour it from the RGB grid.

    Grid convention (``meta.frame``): index ``[x][y][z]``, x right, y up,
    z towards the viewer; voxel ``i`` spans ``[i, i+1) / resolution`` of the
    unit cube. The returned mesh lives in that unit cube.

    The grid is padded with empty voxels first, so a shape touching the
    boundary still yields a closed surface.

    Raises:
        GenerationError: when no voxel crosses ``threshold``.
    """
    import trimesh  # noqa: PLC0415
    from skimage import measure  # noqa: PLC0415

    occupancy = np.asarray(occupancy, dtype=np.float32)
    resolution = occupancy.shape[-1]
    occupancy = occupancy.reshape(resolution, resolution, resolution)
    padded = np.pad(occupancy, 1, mode="constant", constant_values=0.0)
    if float(padded.max()) <= threshold:
        raise _empty_shape_error()

    vertices, faces, _, _ = measure.marching_cubes(padded, level=threshold)
    # Padded index j is voxel j - 1, whose centre sits at (j - 1 + 0.5) / res.
    positions = (vertices - 0.5) / resolution

    colours = None
    if rgb is not None:
        colours = _vertex_colours(padded, rgb, vertices, resolution)

    mesh = trimesh.Trimesh(vertices=positions, faces=faces, vertex_colors=colours,
                           process=True)
    if len(mesh.faces) and mesh.is_watertight and mesh.volume < 0:
        mesh.invert()  # outward-facing normals, whatever skimage's winding
    return mesh


def _vertex_colours(padded_occupancy: np.ndarray, rgb: np.ndarray,
                    vertices: np.ndarray, resolution: int) -> np.ndarray:
    """Per-vertex RGBA, trilinear over the grid and weighted by occupancy.

    The decoder's colour is only trained inside the shape (the loss is masked
    by occupancy), so empty voxels next to the surface carry arbitrary colour.
    Weighting each neighbour by its occupancy keeps the surface colour from
    the solid side.
    """
    from scipy import ndimage  # noqa: PLC0415

    grid = np.asarray(rgb, dtype=np.float32).reshape(3, resolution, resolution, resolution)
    coordinates = vertices.T
    weight = ndimage.map_coordinates(padded_occupancy, coordinates, order=1, mode="nearest")
    colours = np.empty((len(vertices), 3), dtype=np.float64)
    for channel in range(3):
        padded = np.pad(grid[channel], 1, mode="edge")
        weighted = ndimage.map_coordinates(padded_occupancy * padded, coordinates,
                                           order=1, mode="nearest")
        plain = ndimage.map_coordinates(padded, coordinates, order=1, mode="nearest")
        colours[:, channel] = np.where(weight > 1e-4, weighted / np.maximum(weight, 1e-4),
                                       plain)
    rgba = np.empty((len(vertices), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(np.rint(np.clip(colours, 0.0, 1.0) * 255.0), 0, 255)
    rgba[:, 3] = 255
    return rgba


def scale_to_spec(mesh, spec: DesignSpec | None) -> dict[str, Any]:
    """Place the mesh on the ground and scale it uniformly to the specification.

    Centred on x/z with its base at y = 0. Scaling is uniform - the shape is
    never distorted to hit a number:

    * a height is given        -> the y extent equals it;
    * else a length/width      -> the larger of the x/z extents equals the
                                  larger of the two;
    * else                     -> the largest extent is 1 m.

    Units are metres, the glTF convention the rest of the pipeline uses.
    """
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    mesh.apply_translation([
        -(bounds[0][0] + bounds[1][0]) / 2.0,
        -bounds[0][1],
        -(bounds[0][2] + bounds[1][2]) / 2.0,
    ])
    extents = np.maximum(np.asarray(mesh.extents, dtype=np.float64), 1e-9)

    dimensions = spec.dimensions if spec is not None else None
    footprint = [value for value in (
        dimensions.length_mm if dimensions else None,
        dimensions.width_mm if dimensions else None,
    ) if value]
    if dimensions is not None and dimensions.height_mm:
        rule = "height"
        factor = dimensions.height_mm / 1000.0 / extents[1]
    elif footprint:
        rule = "footprint"
        factor = max(footprint) / 1000.0 / max(extents[0], extents[2])
    else:
        rule = "default (largest extent 1 m)"
        factor = 1.0 / float(extents.max())
    mesh.apply_scale(float(factor))

    size = np.asarray(mesh.extents, dtype=np.float64)
    return {
        "rule": rule,
        "factor": round(float(factor), 6),
        "width_m": round(float(size[0]), 6),
        "height_m": round(float(size[1]), 6),
        "depth_m": round(float(size[2]), 6),
    }


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


@dataclass
class _Sessions:
    """Everything loaded from disk, shared across generations."""

    meta: dict[str, Any]
    tokenizer: Any
    text: Any
    text_inputs: tuple[str, ...]
    prior: Any
    decoder: Any
    decoder_outputs: tuple[str, ...]
    #: Modification times of the files, so a reinstall is picked up.
    stamp: tuple[float, ...]


def read_meta(model_dir: Path) -> dict[str, Any] | None:
    """Parse ``meta.json``, or ``None`` when it is absent or unreadable."""
    try:
        return json.loads((Path(model_dir) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Text2VoxelProvider(TextTo3DGenerator):
    """Prompt -> coloured mesh with the Text2Voxel-64 latent diffusion model."""

    name = "text2voxel"
    label = "Text2Voxel-64 (text-to-3D, local)"

    def __init__(self, model_dir: str | Path, steps: int | None = None,
                 guidance: float | None = None, threads: int | None = None) -> None:
        self.model_dir = Path(model_dir)
        #: ``None`` means "use the value the model was trained with" (meta.json).
        self.steps = steps if steps and steps > 0 else None
        self.guidance = guidance
        self.threads = threads
        self._sessions: _Sessions | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ capability
    def missing_files(self) -> list[str]:
        return [name for name in MODEL_FILES if not (self.model_dir / name).is_file()]

    def _stamp(self) -> tuple[float, ...]:
        stamps = []
        for name in MODEL_FILES:
            try:
                stamps.append((self.model_dir / name).stat().st_mtime)
            except OSError:
                stamps.append(-1.0)
        return tuple(stamps)

    def availability(self) -> Availability:
        for module, package in REQUIRED_PACKAGES:
            try:
                importlib.import_module(module)
            except ImportError:
                return Availability.missing(
                    f"The '{package}' package is not installed.",
                    requires=[package], experimental=True,
                )
        missing = self.missing_files()
        if missing:
            return Availability.missing(
                f"The Text2Voxel-64 model files are not installed "
                f"(missing: {', '.join(missing)}). " + INSTALL_HINT,
                requires=["text2voxel model files"], experimental=True,
            )
        meta = read_meta(self.model_dir)
        if meta is None:
            return Availability.missing(
                "The Text2Voxel-64 meta.json could not be read. "
                + INSTALL_HINT + " --force",
                requires=["text2voxel model files"], experimental=True,
            )
        if meta.get("smoke"):
            return Availability(
                available=True, experimental=True,
                reason="The installed Text2Voxel-64 files are the untrained placeholder "
                       "(meta.json has \"smoke\": true), so the shapes it produces are "
                       "meaningless. Fetch the trained release with: "
                       "python scripts/install_text2voxel.py --force",
            )
        return Availability.ok(experimental=True)

    # --------------------------------------------------------------- loading
    def _load(self) -> _Sessions:
        """Open the three ONNX sessions and the tokenizer. Caller holds the lock.

        The sessions are reused across generations. If the files change on
        disk (``install_text2voxel.py --force`` swapping the placeholder for
        the trained release), they are reloaded rather than served stale.
        """
        if self._sessions is not None:
            if self._sessions.stamp == self._stamp():
                return self._sessions
            logger.info("Text2Voxel-64 files changed on disk - reloading")
            self._sessions = None

        availability = self.availability()
        if not availability.available:
            raise ProviderUnavailableError(availability.reason, hint=INSTALL_HINT)

        import onnxruntime as ort  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        started = time.perf_counter()
        stamp = self._stamp()
        try:
            meta = read_meta(self.model_dir) or {}
            diffusion = meta["diffusion"]
            if len(diffusion["alphas_cumprod"]) != int(diffusion["train_steps"]):
                raise ValueError("alphas_cumprod does not match train_steps")

            tokenizer = Tokenizer.from_file(str(self.model_dir / "minilm" / "tokenizer.json"))
            tokenizer.no_padding()
            max_tokens = int(meta.get("text_encoder", {}).get("max_tokens", 128))
            tokenizer.enable_truncation(max_tokens)

            options = ort.SessionOptions()
            if self.threads:
                options.intra_op_num_threads = int(self.threads)
            cpu = ["CPUExecutionProvider"]

            def session(relative: str):
                return ort.InferenceSession(str(self.model_dir / relative), options,
                                            providers=cpu)

            text = session("minilm/model_quantized.onnx")
            prior = session("prior.onnx")
            decoder = session("decoder.onnx")
        except Exception as exc:
            raise ProviderUnavailableError(
                "The Text2Voxel-64 model files could not be loaded.",
                hint="They may be incomplete or corrupt. " + INSTALL_HINT + " --force",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        self._sessions = _Sessions(
            meta=meta,
            tokenizer=tokenizer,
            text=text,
            text_inputs=tuple(item.name for item in text.get_inputs()),
            prior=prior,
            decoder=decoder,
            decoder_outputs=tuple(item.name for item in decoder.get_outputs()),
            stamp=stamp,
        )
        logger.info(
            "Loaded %s v%s%s in %.2fs", meta.get("name", "Text2Voxel"),
            meta.get("version", "?"), " (placeholder)" if meta.get("smoke") else "",
            time.perf_counter() - started,
        )
        return self._sessions

    def release(self) -> None:
        with self._lock:
            if self._sessions is None:
                return
            self._sessions = None
            logger.info("Released Text2Voxel-64 sessions")

    # ---------------------------------------------------------- model steps
    @staticmethod
    def _run(session, feeds: dict[str, np.ndarray], what: str) -> list[np.ndarray]:
        try:
            return session.run(None, feeds)
        except Exception as exc:  # onnxruntime raises its own exception types
            raise GenerationError(
                f"Text2Voxel-64 failed while running the {what}.",
                hint="The model files may not match this version of the app. "
                     + INSTALL_HINT + " --force",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _embed(self, sessions: _Sessions, prompt: str) -> np.ndarray:
        encoding = sessions.tokenizer.encode(prompt)
        ids = np.asarray([encoding.ids], dtype=np.int64)
        mask = np.ones_like(ids)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in sessions.text_inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._run(sessions.text, feeds, "text encoder")[0]
        return mean_pool(hidden, mask)[0]

    def _prior_fn(self, sessions: _Sessions) -> PriorFn:
        def prior(x: np.ndarray, t: np.ndarray, cond: np.ndarray) -> np.ndarray:
            feeds = {"x": x.astype(np.float32), "t": t.astype(np.float32),
                     "cond": cond.astype(np.float32)}
            return self._run(sessions.prior, feeds, "diffusion prior")[0]

        return prior

    def _decode(self, sessions: _Sessions, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = self._run(sessions.decoder,
                            {"z": latent.reshape(1, -1).astype(np.float32)}, "decoder")
        named = dict(zip(sessions.decoder_outputs, outputs, strict=False))
        occupancy = named.get("occupancy", outputs[0])
        rgb = named.get("rgb", outputs[1] if len(outputs) > 1 else None)
        return occupancy[0], (rgb[0] if rgb is not None else None)

    def _settings(self, meta: dict[str, Any], steps: int | None,
                  guidance: float | None) -> tuple[int, float]:
        diffusion = meta["diffusion"]
        chosen_steps = int(steps or self.steps or diffusion.get("sample_steps", 50))
        chosen_steps = max(1, min(chosen_steps, int(diffusion["train_steps"])))
        chosen_guidance = guidance if guidance is not None else self.guidance
        if chosen_guidance is None:
            chosen_guidance = float(diffusion.get("guidance", 3.0))
        return chosen_steps, float(chosen_guidance)

    # ------------------------------------------------------------ public API
    def embed(self, prompt: str) -> np.ndarray:
        """The 384-d conditioning vector for ``prompt``."""
        with self._lock:
            return self._embed(self._load(), prompt)

    def sample_latent(self, prompt: str, noise: np.ndarray, *, steps: int | None = None,
                      guidance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Embed ``prompt`` and run the sampler from ``noise``; returns (cond, latent)."""
        with self._lock:
            sessions = self._load()
            chosen_steps, chosen_guidance = self._settings(sessions.meta, steps, guidance)
            cond = self._embed(sessions, prompt)
            diffusion = sessions.meta["diffusion"]
            latent = ddim_sample(
                self._prior_fn(sessions), cond, noise, diffusion["alphas_cumprod"],
                steps=chosen_steps, guidance=chosen_guidance,
                x0_clip=float(diffusion.get("x0_clip", 6.0)),
            )
            return cond, latent

    def generate_from_text(
        self,
        prompt: str,
        output_dir: Path,
        *,
        spec: DesignSpec | None = None,
        seed: int | None = None,
        progress: ProgressCallback = _noop_progress,
        steps: int | None = None,
        guidance: float | None = None,
    ) -> ModelResult:
        prompt = (prompt or "").strip()
        if not prompt:
            raise GenerationError("Text2Voxel-64 needs a prompt to generate from.")

        output_dir.mkdir(parents=True, exist_ok=True)
        seed = int(seed) if seed is not None else random.SystemRandom().randrange(2**31 - 1)
        started = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(key: str, since: float) -> float:
            now = time.perf_counter()
            timings[key] = round((now - since) * 1000.0, 1)
            return now

        with self._lock:
            progress(0.02, "Loading Text2Voxel-64")
            mark = time.perf_counter()
            sessions = self._load()
            meta = sessions.meta
            diffusion = meta["diffusion"]
            chosen_steps, chosen_guidance = self._settings(meta, steps, guidance)
            mark = lap("load", mark)

            progress(0.08, "Encoding the prompt")
            cond = self._embed(sessions, prompt)
            mark = lap("encode", mark)

            latent_dim = int(meta.get("latent_dim", 256))
            noise = np.random.default_rng(seed).standard_normal(latent_dim).astype(np.float32)

            def on_step(index: int, total: int) -> None:
                # The job's progress callback raises JobCancelled once the user
                # cancels, so a cancelled job stops here between steps.
                fraction = SAMPLING_START + (SAMPLING_END - SAMPLING_START) * (index + 1) / total
                progress(fraction, f"Sampling step {index + 1}/{total}")

            progress(SAMPLING_START, f"Sampling the shape latent ({chosen_steps} steps)")
            latent = ddim_sample(
                self._prior_fn(sessions), cond, noise, diffusion["alphas_cumprod"],
                steps=chosen_steps, guidance=chosen_guidance,
                x0_clip=float(diffusion.get("x0_clip", 6.0)), on_step=on_step,
            )
            mark = lap("sample", mark)

            progress(0.84, "Decoding the voxel grid")
            occupancy, rgb = self._decode(sessions, latent)
            mark = lap("decode", mark)

        threshold = float(meta.get("threshold", 0.5))
        shape = prepare_field(occupancy, threshold)
        occupied = shape.occupied_voxels
        progress(0.9, f"Extracting the surface ({occupied:,} occupied voxels)")
        # One float32 step below the iso-level, so a voxel exactly at it counts
        # as inside (the browser's ">= iso") and no vertex lands on a grid point.
        level = float(np.nextafter(np.float32(shape.iso), np.float32(0.0)))
        mesh = voxels_to_mesh(shape.field, rgb, level)
        if mesh.is_empty or len(mesh.faces) == 0:
            raise GenerationError("Text2Voxel-64 produced an empty mesh.",
                                  hint="Try another seed.")
        scale = scale_to_spec(mesh, spec)
        mark = lap("mesh", mark)

        progress(0.96, "Writing GLB")
        model_path = output_dir / "model.glb"
        mesh.export(model_path)
        lap("write", mark)

        elapsed = time.perf_counter() - started
        smoke = bool(meta.get("smoke", False))
        progress(1.0, f"Generated {len(mesh.faces):,} triangles in {elapsed:.1f}s")
        return ModelResult(
            path=model_path,
            file_format="glb",
            provider=self.name,
            generation_time_s=elapsed,
            metadata={
                "method": "text-to-3D latent diffusion (Text2Voxel-64)",
                "model": meta.get("name", "Text2Voxel-64"),
                "model_version": meta.get("version"),
                "model_created": meta.get("created"),
                "smoke": smoke,
                "warning": "Placeholder model (untrained smoke-test weights) - the shape "
                           "is meaningless." if smoke else None,
                "prompt": prompt,
                "seed": seed,
                "noise": "numpy default_rng(seed).standard_normal",
                "guidance": chosen_guidance,
                "steps": chosen_steps,
                "resolution": int(meta.get("resolution", occupancy.shape[-1])),
                "threshold": threshold,
                "iso_level": shape.iso,
                "occupied_voxels": occupied,
                "warnings": shape.warnings,
                "timings_ms": timings,
                "scale": scale,
                # The model's output is already in the canonical frame (front
                # faces +Z), so mesh processing must not re-orient it.
                "canonical_frame": True,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "is_watertight": bool(mesh.is_watertight),
            },
        )
