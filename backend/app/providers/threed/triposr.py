"""Learned image-to-3D reconstruction with TripoSR.

TripoSR is a large reconstruction model: it predicts a triplane NeRF from a
single image in one forward pass, then marches a mesh out of the density
field. It is both faster and sharper than Shap-E - roughly 9 s versus 90 s on
a laptop GPU, with far more surface detail.

Two compatibility layers live here because upstream TripoSR is pinned to a
2024 environment:

* **Marching cubes.** TripoSR imports ``torchmcubes``, a source-only CUDA
  extension needing a full build toolchain. scikit-image already provides
  marching cubes, so :func:`_install_marching_cubes_shim` registers an
  equivalent module instead. No compiler required.
* **ViT weights.** The checkpoint was saved when transformers named ViT blocks
  ``encoder.layer.N.attention.attention.query``; current versions use
  ``layers.N.attention.q_proj``. The tensors are identical, only the paths
  moved, so :func:`_remap_vit_keys` renames them. Verified to load with zero
  missing and zero unexpected keys.

TripoSR itself is not on PyPI and is not vendored here. Install it with::

    python scripts/install_triposr.py

Quality note: reconstruction depends heavily on the input image. Matte,
evenly-lit subjects reconstruct well. Dark glossy subjects - a black car with
strong specular highlights - give the model little shading to read depth from
and tend to flatten.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
import types
from functools import lru_cache
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

MODEL_ID = "stabilityai/TripoSR"
INSTALL_HINT = "Run: python scripts/install_triposr.py (from the backend directory)"

#: TripoSR conditions on a single image; a three-quarter view shows the most.
PREFERRED_VIEWS = (
    ViewName.PERSPECTIVE,
    ViewName.FRONT,
    ViewName.LEFT,
    ViewName.RIGHT,
    ViewName.REAR,
    ViewName.TOP,
)

#: Fraction of the frame the cut-out subject should fill, per TripoSR's README.
FOREGROUND_RATIO = 0.85

# --------------------------------------------------------------------------
# Compatibility shims
# --------------------------------------------------------------------------


def _shim_marching_cubes(volume, threshold: float = 0.0):
    """``torchmcubes.marching_cubes`` implemented with scikit-image."""
    import torch  # noqa: PLC0415
    from skimage import measure  # noqa: PLC0415

    array = (volume.detach().cpu().numpy() if isinstance(volume, torch.Tensor)
             else np.asarray(volume))
    try:
        vertices, faces, _, _ = measure.marching_cubes(array, level=threshold)
    except (ValueError, RuntimeError):
        # No surface at this level - an empty mesh is the correct answer.
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.long)

    # torchmcubes yields vertices in reversed index order; TripoSR reverses
    # them again straight after the call, so pre-reverse here to match.
    return (
        torch.from_numpy(np.ascontiguousarray(vertices[:, [2, 1, 0]])).float(),
        torch.from_numpy(np.ascontiguousarray(faces)).long(),
    )


def _install_marching_cubes_shim() -> None:
    """Register the shim under the name TripoSR imports."""
    if "torchmcubes" in sys.modules:
        return
    module = types.ModuleType("torchmcubes")
    module.marching_cubes = _shim_marching_cubes
    module.grid_interp = None
    sys.modules["torchmcubes"] = module


_VIT_LAYER = re.compile(r"^(?P<head>.*?)encoder\.layer\.(?P<index>\d+)\.(?P<rest>.+)$")

#: Old block-local path -> current path.
_VIT_RENAMES = (
    ("attention.attention.query", "attention.q_proj"),
    ("attention.attention.key", "attention.k_proj"),
    ("attention.attention.value", "attention.v_proj"),
    ("attention.output.dense", "attention.o_proj"),
    ("intermediate.dense", "mlp.fc1"),
    ("output.dense", "mlp.fc2"),
)


def _remap_vit_keys(state_dict: dict) -> tuple[dict, int]:
    """Rename transformers<=4.35 ViT keys to the current layout."""
    remapped: dict = {}
    changed = 0
    for key, value in state_dict.items():
        match = _VIT_LAYER.match(key)
        if not match:
            remapped[key] = value
            continue
        rest = match["rest"]
        for old, new in _VIT_RENAMES:
            if rest.startswith(old + "."):
                rest = new + rest[len(old):]
                break
        remapped[f"{match['head']}layers.{match['index']}.{rest}"] = value
        changed += 1
    return remapped, changed


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _source_available(source: str) -> bool:
    return bool(source) and (Path(source) / "tsr" / "system.py").is_file()


class TripoSRProvider(ThreeDGenerator):
    """Single-image learned image-to-3D via TripoSR."""

    name = "triposr"
    label = "TripoSR image-to-3D (local, learned)"

    def __init__(self, source_path: str, resolution: int = 256,
                 device: str = "auto", chunk_size: int = 8192) -> None:
        self.source_path = source_path
        self.resolution = int(np.clip(resolution, 64, 512))
        self._configured_device = device
        self.chunk_size = chunk_size
        self._model = None
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
            return Availability.missing("PyTorch is not installed.",
                                        requires=["torch"], experimental=True)
        for module in ("omegaconf", "einops"):
            try:
                __import__(module)
            except ImportError:
                return Availability.missing(
                    f"The '{module}' package is required by TripoSR.",
                    requires=[module], experimental=True,
                )
        if not _source_available(self.source_path):
            return Availability.missing(
                "The TripoSR source was not found. " + INSTALL_HINT,
                requires=["TripoSR"], experimental=True,
            )
        if self.resolve_device() == "cpu":
            return Availability(
                available=True, experimental=True,
                reason="Running on CPU - a reconstruction takes several minutes.",
            )
        return Availability.ok(experimental=True)

    # --------------------------------------------------------------- loading
    def _load(self):
        """Build the model and load remapped weights. Caller holds the lock."""
        if self._model is not None:
            return self._model

        availability = self.availability()
        if not availability.available:
            raise ProviderUnavailableError(availability.reason, hint=INSTALL_HINT)

        import torch  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        _install_marching_cubes_shim()
        if self.source_path not in sys.path:
            sys.path.insert(0, self.source_path)
        from tsr.system import TSR  # noqa: PLC0415

        try:
            config_path = hf_hub_download(MODEL_ID, "config.yaml")
            weight_path = hf_hub_download(MODEL_ID, "model.ckpt")
        except Exception as exc:
            raise ProviderUnavailableError(
                "The TripoSR weights could not be downloaded or found in the cache.",
                hint="Check your network connection, or switch the 3D provider to "
                     "'visual_hull' in Settings.",
                detail=str(exc),
            ) from exc

        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        model = TSR(config)

        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        state_dict, changed = _remap_vit_keys(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ProviderUnavailableError(
                "The TripoSR checkpoint does not match this build of transformers.",
                hint="Pin transformers to the version in TripoSR's requirements, or "
                     "use the 'shap_e' provider instead.",
                detail=f"{len(missing)} missing, {len(unexpected)} unexpected keys",
            )
        logger.info("Loaded TripoSR (remapped %d ViT keys)", changed)

        model.renderer.set_chunk_size(self.chunk_size)
        self._model = model.to(self.resolve_device()).eval()
        return self._model

    def release(self) -> None:
        with self._lock:
            if self._model is None:
                return
            self._model = None
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass
            logger.info("Released TripoSR")

    # -------------------------------------------------------- preprocessing
    @staticmethod
    def _prepare(image_path: Path):
        """Cut the subject out, centre it, and composite onto neutral grey.

        TripoSR is trained on subjects isolated this way; feeding it a full
        studio photograph (background included) reconstructs the whole frame
        and yields a featureless blob.
        """
        from PIL import Image  # noqa: PLC0415

        from app.providers.threed.visual_hull import _matting_session  # noqa: PLC0415

        with Image.open(image_path) as opened:
            source = opened.convert("RGB")

        session = _matting_session()
        if session is None:
            raise ProviderUnavailableError(
                "TripoSR needs background removal, which requires the 'rembg' model.",
                hint="Install rembg and let its model download, or use the "
                     "'visual_hull' provider.",
            )

        from rembg import remove  # noqa: PLC0415

        cut_out = remove(source, session=session).convert("RGBA")
        array = np.asarray(cut_out)
        alpha = np.where(array[..., 3] > 0)
        if alpha[0].size == 0:
            raise GenerationError(
                "The subject could not be separated from the background.",
                hint="Regenerate the view, or upload a reference image with a "
                     "clean background.",
            )

        # Crop to the subject, pad to a square, then inset by the trained ratio.
        top, bottom = alpha[0].min(), alpha[0].max()
        left, right = alpha[1].min(), alpha[1].max()
        cropped = array[top:bottom + 1, left:right + 1]
        size = max(cropped.shape[0], cropped.shape[1])
        pad_y = (size - cropped.shape[0]) // 2
        pad_x = (size - cropped.shape[1]) // 2
        square = np.zeros((size, size, 4), dtype=array.dtype)
        square[pad_y:pad_y + cropped.shape[0], pad_x:pad_x + cropped.shape[1]] = cropped

        margin = int(size * (1 / FOREGROUND_RATIO - 1) / 2)
        padded = np.pad(square, ((margin, margin), (margin, margin), (0, 0)))

        rgba = padded.astype(np.float32) / 255.0
        composited = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4]) * 0.5
        return Image.fromarray((composited * 255).astype(np.uint8)).resize((512, 512))

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

        if not images:
            raise GenerationError("No images are available to reconstruct from.")

        output_dir.mkdir(parents=True, exist_ok=True)
        chosen = self._pick_image(images)
        started = time.time()

        with self._lock:
            progress(0.05, "Loading TripoSR")
            model = self._load()
            device = self.resolve_device()

            progress(0.2, f"Preparing the {chosen.view.value} view")
            prepared = self._prepare(chosen.path)

            progress(0.35, "Reconstructing")
            try:
                with torch.no_grad():
                    codes = model([prepared], device=device)
                progress(0.7, "Extracting surface")
                mesh_out = model.extract_mesh(codes, True, resolution=self.resolution)[0]
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise OutOfMemoryError(detail=str(exc)) from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise OutOfMemoryError(detail=str(exc)) from exc
                raise GenerationError("TripoSR failed during reconstruction.",
                                      detail=str(exc)) from exc

        vertices = np.asarray(mesh_out.vertices, dtype=np.float64)
        faces = np.asarray(mesh_out.faces, dtype=np.int64)
        if len(faces) == 0:
            raise GenerationError(
                "TripoSR returned an empty mesh.",
                hint="Try a different source view, or the 'visual_hull' provider.",
            )

        # TripoSR's frame is x back, y right, z up. glTF is x right, y up,
        # z towards the viewer.
        vertices = np.stack(
            [vertices[:, 1], vertices[:, 2], -vertices[:, 0]], axis=1
        )

        colours = getattr(mesh_out.visual, "vertex_colors", None)
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=colours, process=True
        )
        if mesh.is_empty or len(mesh.faces) == 0:
            raise GenerationError("TripoSR produced an empty mesh.")

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
                "method": "learned image-to-3D (TripoSR)",
                "model": MODEL_ID,
                "source_view": chosen.view.value,
                "marching_cubes_resolution": self.resolution,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            },
        )
