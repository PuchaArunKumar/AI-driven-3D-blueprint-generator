"""Local Stable Diffusion image provider backed by :mod:`diffusers`.

Runs entirely on the machine hosting the backend - no API key, no network
access once the weights are cached.  The pipeline is loaded lazily on first
use and kept in memory afterwards; :meth:`release` frees it and empties the
CUDA cache.

Multi-view consistency is approached the way the research pipeline describes
it: one shared identity clause per project (see
:func:`app.pipeline.prompt_parser.spec_to_base_prompt`) plus a deterministic
per-view seed derived from a single project seed.  This keeps colour, material
and proportion broadly stable across views.  It is *not* a true multi-view
diffusion model, and the UI says so.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from app.errors import GenerationError, OutOfMemoryError, ProviderUnavailableError
from app.pipeline.prompt_parser import NEGATIVE_PROMPT, build_view_prompt
from app.providers.base import (
    Availability,
    ImageGenerator,
    ImageResult,
    ProgressCallback,
    _noop_progress,
)
from app.schemas import DesignSpec, ViewName

logger = logging.getLogger(__name__)

#: Models that are distilled for very low step counts and ignore guidance.
TURBO_MODELS = ("sd-turbo", "sdxl-turbo", "lcm", "lightning")


def _torch():
    """Import torch lazily so the app starts without it installed."""
    import torch  # noqa: PLC0415

    return torch


class DiffusersImageProvider(ImageGenerator):
    """Stable Diffusion running locally through the ``diffusers`` library."""

    name = "diffusers"
    label = "Stable Diffusion (local)"

    def __init__(
        self,
        model_id: str = "stabilityai/sd-turbo",
        device: str = "auto",
        steps: int = 4,
        guidance: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self._configured_device = device
        self.steps = max(1, steps)
        self.guidance = guidance
        self._pipe = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ capability
    @staticmethod
    def _import_diffusers():
        try:
            import diffusers  # noqa: PLC0415

            return diffusers
        except ImportError:
            return None

    def resolve_device(self) -> str:
        """Pick the device to run on, honouring an explicit override."""
        if self._configured_device in ("cuda", "cpu"):
            return self._configured_device
        try:
            torch = _torch()
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def availability(self) -> Availability:
        try:
            _torch()
        except ImportError:
            return Availability.missing(
                "PyTorch is not installed in the backend environment.",
                requires=["torch"],
            )
        if self._import_diffusers() is None:
            return Availability.missing(
                "The 'diffusers' package is not installed.",
                requires=["diffusers"],
            )
        if self.resolve_device() == "cpu":
            return Availability(
                available=True,
                reason=(
                    "Running on CPU - a single view can take several minutes. "
                    "Install a CUDA build of PyTorch for GPU acceleration."
                ),
            )
        return Availability.ok()

    # --------------------------------------------------------------- loading
    def _load(self):
        """Load and cache the pipeline. Callers must hold ``self._lock``."""
        if self._pipe is not None:
            return self._pipe

        diffusers = self._import_diffusers()
        if diffusers is None:
            raise ProviderUnavailableError(
                "Local image generation needs the 'diffusers' package.",
                hint="Run: pip install -r backend/requirements.txt",
            )
        torch = _torch()
        device = self.resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading diffusers pipeline %s on %s (%s)", self.model_id, device, dtype)

        # On CUDA the weights are cast to fp16 anyway, so prefer the fp16
        # variant: it is roughly half the download and half the disk. Not every
        # checkpoint publishes one, so fall back to the default weights.
        variants: list[str | None] = ["fp16", None] if device == "cuda" else [None]
        pipe = None
        last_error: Exception | None = None
        for variant in variants:
            try:
                pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    variant=variant,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                break
            except (OSError, ValueError) as exc:
                last_error = exc
                if variant is not None:
                    logger.info("No '%s' variant for %s, falling back to the default weights",
                                variant, self.model_id)

        if pipe is None:
            raise ProviderUnavailableError(
                f"The model '{self.model_id}' could not be downloaded or found in the cache.",
                hint="Check your network connection and available disk space. The demo "
                     "projects in the Gallery work offline with no provider configured.",
                detail=str(last_error),
            ) from last_error

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        if device == "cuda":
            # Keeps peak VRAM within reach of a 6 GB laptop GPU.
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
        self._pipe = pipe
        return pipe

    def release(self) -> None:
        with self._lock:
            if self._pipe is None:
                return
            self._pipe = None
            try:
                torch = _torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - torch always present here
                pass
            logger.info("Released diffusers pipeline")

    # ------------------------------------------------------------ generation
    def _is_turbo(self) -> bool:
        lowered = self.model_id.lower()
        return any(token in lowered for token in TURBO_MODELS)

    def generate(
        self,
        spec: DesignSpec,
        views: list[ViewName],
        output_dir: Path,
        *,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
        progress: ProgressCallback = _noop_progress,
    ) -> list[ImageResult]:
        torch = _torch()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Diffusion UNets require both dimensions to be multiples of 8.
        width, height = (max(256, (value // 8) * 8) for value in (width, height))

        with self._lock:
            pipe = self._load()
            device = self.resolve_device()
            base_seed = seed if seed is not None else int(time.time()) % (2**31 - 1)
            turbo = self._is_turbo()
            results: list[ImageResult] = []

            for index, view in enumerate(views):
                progress(index / len(views), f"Rendering {view.value} view")
                view_seed = (base_seed + index * 1013) % (2**31 - 1)
                generator = torch.Generator(device=device).manual_seed(view_seed)
                kwargs: dict[str, object] = {
                    "prompt": build_view_prompt(spec, view),
                    "num_inference_steps": self.steps,
                    "guidance_scale": self.guidance,
                    "width": width,
                    "height": height,
                    "generator": generator,
                }
                # Turbo/LCM checkpoints are trained without classifier-free
                # guidance, so a negative prompt has no effect and diffusers
                # rejects it when guidance_scale is 0.
                if not turbo:
                    kwargs["negative_prompt"] = NEGATIVE_PROMPT

                try:
                    image = pipe(**kwargs).images[0]
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise OutOfMemoryError(detail=str(exc)) from exc
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "out of memory" in message:
                        raise OutOfMemoryError(detail=str(exc)) from exc
                    raise GenerationError(
                        "Image generation failed inside the diffusion pipeline.",
                        detail=str(exc),
                    ) from exc

                path = output_dir / f"{view.value}.png"
                image.save(path)
                results.append(
                    ImageResult(
                        view=view,
                        path=path,
                        width=image.width,
                        height=image.height,
                        provider=self.name,
                        seed=view_seed,
                    )
                )

            progress(1.0, f"Rendered {len(results)} views")
            return results
