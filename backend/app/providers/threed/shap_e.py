"""Learned image-to-3D reconstruction with Shap-E.

Shap-E is a generative 3D model: it decodes a latent into an implicit function
and marches a mesh out of it.  Unlike the visual hull it has a *prior* over what
objects look like, so it produces plausible geometry where no view gives
evidence - wheel wells, a hollowed cockpit, the underside of a seat.

That prior is also the trade-off, and the reason both providers ship:

    visual_hull  - faithful to the generated views, blind to concavities,
                   fails when the views are not true orthographic elevations.
    shap_e       - recovers object-like structure from a single view, but the
                   geometry is the model's idea of the object rather than a
                   measurement of the views. Resolution is modest.

Runs locally on GPU or CPU; the weights (~1.3 GB) download once.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from app.errors import GenerationError, OutOfMemoryError, ProviderUnavailableError
from app.providers.base import (
    Availability,
    ImageResult,
    ModelResult,
    ProgressCallback,
    ThreeDGenerator,
    _noop_progress,
)
from app.schemas import DesignSpec, ViewName

logger = logging.getLogger(__name__)

MODEL_ID = "openai/shap-e-img2img"

#: Shap-E conditions on one image. A three-quarter view carries the most shape
#: information; a flat elevation hides a whole axis.
PREFERRED_VIEWS = (
    ViewName.PERSPECTIVE,
    ViewName.FRONT,
    ViewName.LEFT,
    ViewName.RIGHT,
    ViewName.REAR,
    ViewName.TOP,
)


class ShapEProvider(ThreeDGenerator):
    """Single-image learned image-to-3D via Shap-E."""

    name = "shap_e"
    label = "Shap-E image-to-3D (local, learned)"

    def __init__(self, steps: int = 64, guidance: float = 3.0,
                 device: str = "auto") -> None:
        self.steps = max(8, steps)
        self.guidance = guidance
        self._configured_device = device
        self._pipe = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ capability
    def resolve_device(self) -> str:
        if self._configured_device in ("cuda", "cpu"):
            return self._configured_device
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def availability(self) -> Availability:
        try:
            import torch  # noqa: F401,PLC0415
        except ImportError:
            return Availability.missing(
                "PyTorch is not installed in the backend environment.",
                requires=["torch"], experimental=True,
            )
        try:
            import diffusers  # noqa: PLC0415
        except ImportError:
            return Availability.missing(
                "The 'diffusers' package is not installed.",
                requires=["diffusers"], experimental=True,
            )
        if not hasattr(diffusers, "ShapEImg2ImgPipeline"):
            return Availability.missing(
                "This diffusers version does not provide ShapEImg2ImgPipeline.",
                requires=["diffusers>=0.18"], experimental=True,
            )
        if self.resolve_device() == "cpu":
            return Availability(
                available=True,
                reason="Running on CPU - a single reconstruction takes several minutes.",
                experimental=True,
            )
        return Availability.ok(experimental=True)

    # --------------------------------------------------------------- loading
    def _load(self):
        """Load and cache the pipeline. Callers must hold ``self._lock``."""
        if self._pipe is not None:
            return self._pipe

        import torch  # noqa: PLC0415
        from diffusers import ShapEImg2ImgPipeline  # noqa: PLC0415

        device = self.resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading Shap-E (%s) on %s", MODEL_ID, device)
        try:
            pipe = ShapEImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        except OSError as exc:
            raise ProviderUnavailableError(
                "The Shap-E weights could not be downloaded or found in the cache.",
                hint="Check your network connection, or switch the 3D provider back "
                     "to 'visual_hull' in Settings.",
                detail=str(exc),
            ) from exc

        self._pipe = pipe.to(device)
        return self._pipe

    def release(self) -> None:
        with self._lock:
            if self._pipe is None:
                return
            self._pipe = None
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass
            logger.info("Released Shap-E pipeline")

    # ------------------------------------------------------------ generation
    @staticmethod
    def _pick_image(images: list[ImageResult]) -> ImageResult:
        by_view = {image.view: image for image in images}
        for view in PREFERRED_VIEWS:
            if view in by_view:
                return by_view[view]
        return images[0]

    def generate(
        self,
        images: list[ImageResult],
        output_dir: Path,
        *,
        spec: DesignSpec | None = None,
        resolution: int = 128,
        progress: ProgressCallback = _noop_progress,
    ) -> ModelResult:
        import torch  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        if not images:
            raise GenerationError("No images are available to reconstruct from.")

        output_dir.mkdir(parents=True, exist_ok=True)
        chosen = self._pick_image(images)
        started = time.time()

        with self._lock:
            progress(0.05, "Loading Shap-E")
            pipe = self._load()

            progress(0.2, f"Reconstructing from the {chosen.view.value} view")
            with Image.open(chosen.path) as opened:
                source = opened.convert("RGB").resize((256, 256), Image.LANCZOS)

            try:
                result = pipe(
                    source,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance,
                    frame_size=256,
                    output_type="mesh",
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise OutOfMemoryError(detail=str(exc)) from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise OutOfMemoryError(detail=str(exc)) from exc
                raise GenerationError(
                    "Shap-E failed during reconstruction.", detail=str(exc)
                ) from exc

        progress(0.8, "Building mesh")
        decoded = result.images[0]

        def to_numpy(value) -> np.ndarray:
            """Shap-E returns tensors still on the GPU."""
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        vertices = to_numpy(decoded.verts).astype(np.float64)
        faces = to_numpy(decoded.faces).astype(np.int64)
        if len(faces) == 0:
            raise GenerationError(
                "Shap-E returned an empty mesh.",
                hint="Try a different source view, or switch to the visual_hull provider.",
            )

        # Shap-E emits a Z-up mesh; the rest of the pipeline is glTF Y-up.
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 2] *= -1.0

        colours = getattr(decoded, "vertex_channels", None)
        vertex_colors = None
        if isinstance(colours, dict) and {"R", "G", "B"} <= set(colours):
            vertex_colors = np.clip(
                np.stack([to_numpy(colours[c]) for c in ("R", "G", "B")], axis=1), 0, 1
            )
            vertex_colors = (vertex_colors * 255).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=True
        )
        if mesh.is_empty or len(mesh.faces) == 0:
            raise GenerationError("Shap-E produced an empty mesh.")

        progress(0.9, "Scaling to specification")
        from app.providers.threed.visual_hull import _apply_scale  # noqa: PLC0415

        _apply_scale(mesh, spec)

        model_path = output_dir / "model.glb"
        mesh.export(model_path)

        elapsed = time.time() - started
        progress(1.0, f"Reconstructed {len(mesh.faces):,} triangles in {elapsed:.1f}s")
        return ModelResult(
            path=model_path,
            file_format="glb",
            provider=self.name,
            generation_time_s=elapsed,
            metadata={
                "method": "learned image-to-3D (Shap-E)",
                "model": MODEL_ID,
                "source_view": chosen.view.value,
                "steps": self.steps,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            },
        )
