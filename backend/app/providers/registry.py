"""Provider registry.

Builds provider instances from settings and answers "what can this
installation actually do right now?".  Nothing else in the application
constructs a provider directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Settings, get_settings
from app.errors import ProviderUnavailableError
from app.providers.base import CADExporter, ImageGenerator, MeshProcessor, ThreeDGenerator
from app.providers.cad.blender import BlenderProvider
from app.providers.cad.freecad import FreeCADProvider
from app.providers.image.diffusers_local import DiffusersImageProvider
from app.providers.image.hosted import OpenAIImageProvider, StabilityImageProvider
from app.providers.mesh.trimesh_processor import TrimeshProcessor
from app.providers.threed.hosted import HostedThreeDProvider
from app.providers.threed.shap_e import ShapEProvider
from app.providers.threed.text2voxel import Text2VoxelProvider
from app.providers.threed.triposr import TripoSRProvider
from app.providers.threed.visual_hull import VisualHullProvider
from app.schemas import ProviderInfo

logger = logging.getLogger(__name__)

#: Preference order used when the 3D provider is left on "auto".
#:
#: A learned model goes first: silhouette carving needs true orthographic
#: elevations, and text-to-image models return three-quarter product shots for
#: most subjects, which carve into a near-cube. The hull stays as the
#: always-available fallback.
THREED_PREFERENCE = ("triposr", "shap_e", "visual_hull")

#: The text-conditioned provider "auto" falls back to when there are no views
#: to reconstruct from and no image provider that could render them. It is not
#: in the preference list above: with views available, a reconstruction of
#: those views is what the user asked for.
TEXT_FALLBACK = "text2voxel"

#: Formats produced by trimesh alone - always available.
TRIMESH_FORMATS = ("glb", "gltf", "obj", "ply", "stl")


class ProviderRegistry:
    """Owns one instance of each configured provider."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._image: dict[str, ImageGenerator] = {
            "diffusers": DiffusersImageProvider(
                model_id=settings.diffusers_model,
                device=settings.diffusers_device,
                steps=settings.diffusers_steps,
                guidance=settings.diffusers_guidance,
            ),
            "openai": OpenAIImageProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_image_model,
            ),
            "stability": StabilityImageProvider(
                api_key=settings.stability_api_key,
                model=settings.stability_model,
            ),
        }
        self._threed: dict[str, ThreeDGenerator] = {
            "visual_hull": VisualHullProvider(),
            "triposr": TripoSRProvider(
                source_path=str(settings.triposr_path),
                resolution=settings.triposr_resolution,
                device=settings.diffusers_device,
            ),
            "shap_e": ShapEProvider(
                steps=settings.shap_e_steps,
                guidance=settings.shap_e_guidance,
                device=settings.diffusers_device,
            ),
            "trellis_api": HostedThreeDProvider(
                endpoint=settings.trellis_endpoint,
                api_key=settings.trellis_api_key,
            ),
            "text2voxel": Text2VoxelProvider(
                model_dir=settings.text2voxel_path,
                steps=settings.text2voxel_steps,
                guidance=settings.text2voxel_guidance,
            ),
        }
        self._mesh: MeshProcessor = TrimeshProcessor()
        self._cad: dict[str, CADExporter] = {
            "blender": BlenderProvider(settings.blender_path, settings.blender_timeout),
            "freecad": FreeCADProvider(settings.freecad_path, settings.blender_timeout),
        }

    # ------------------------------------------------------------- accessors
    def image_provider(self, name: str | None = None) -> ImageGenerator:
        key = name or self.settings.image_provider
        provider = self._image.get(key)
        if provider is None:
            raise ProviderUnavailableError(
                f"Unknown image provider '{key}'.",
                hint="Choose one of: " + ", ".join(sorted(self._image)),
            )
        return provider

    def threed_provider(self, name: str | None = None, *,
                        allow_text_fallback: bool = False,
                        image_provider: str | None = None) -> ThreeDGenerator:
        """Resolve a 3D provider by name, or pick one for "auto".

        ``allow_text_fallback`` is set by callers that have no views to
        reconstruct from: "auto" then chooses the text-to-3D provider when the
        image provider that would render the views (``image_provider``, else
        the configured one) cannot run.
        """
        key = name or self.settings.threed_provider
        if key == "auto":
            return self._best_threed_provider(
                allow_text_fallback=allow_text_fallback, image_provider=image_provider
            )

        provider = self._threed.get(key)
        if provider is None:
            raise ProviderUnavailableError(
                f"Unknown 3D provider '{key}'.",
                hint="Choose one of: auto, " + ", ".join(sorted(self._threed)),
            )
        return provider

    def _image_provider_available(self, name: str | None) -> bool:
        provider = self._image.get(name or self.settings.image_provider)
        return provider is not None and provider.availability().available

    def _best_threed_provider(self, *, allow_text_fallback: bool = False,
                              image_provider: str | None = None) -> ThreeDGenerator:
        """Pick the best 3D backend that can actually run right now."""
        if allow_text_fallback and not self._image_provider_available(image_provider):
            text = self._threed[TEXT_FALLBACK]
            if text.availability().available:
                logger.info(
                    "Auto-selected 3D provider: %s (no image provider can render views, "
                    "so the model is generated from the prompt text)", TEXT_FALLBACK,
                )
                return text
        for name in THREED_PREFERENCE:
            provider = self._threed.get(name)
            if provider is not None and provider.availability().available:
                logger.info("Auto-selected 3D provider: %s", name)
                return provider
        # visual_hull needs only numpy/scipy/trimesh, so this is near-unreachable.
        raise ProviderUnavailableError(
            "No 3D reconstruction provider is available.",
            hint="Install the backend requirements, or set THREED_PROVIDER explicitly.",
        )

    def mesh_processor(self) -> MeshProcessor:
        return self._mesh

    def cad_provider(self, name: str) -> CADExporter:
        provider = self._cad.get(name)
        if provider is None:
            raise ProviderUnavailableError(f"Unknown CAD provider '{name}'.")
        return provider

    @property
    def blender(self) -> BlenderProvider:
        return self._cad["blender"]  # type: ignore[return-value]

    @property
    def freecad(self) -> FreeCADProvider:
        return self._cad["freecad"]  # type: ignore[return-value]

    # ---------------------------------------------------------- capabilities
    def available_export_formats(self) -> tuple[list[str], dict[str, str]]:
        """Return ``(available, {unavailable_format: reason})``.

        An export format is only offered when something in this installation
        can actually produce it.
        """
        available = list(TRIMESH_FORMATS)
        unavailable: dict[str, str] = {}

        blender = self.blender.availability()
        if blender.available:
            available.append("blend")
        else:
            unavailable["blend"] = blender.reason

        freecad = self.freecad.availability()
        if freecad.available:
            available.extend(("step", "stp"))
        else:
            unavailable["step"] = freecad.reason
            unavailable["stp"] = freecad.reason

        return available, unavailable

    def describe(self) -> list[ProviderInfo]:
        """Availability report for every provider, for the Settings page."""
        infos: list[ProviderInfo] = []
        groups: list[tuple[str, dict[str, object]]] = [
            ("image", self._image),
            ("threed", self._threed),
            ("cad", self._cad),
        ]
        for kind, group in groups:
            for name, provider in group.items():
                availability = provider.availability()  # type: ignore[attr-defined]
                infos.append(
                    ProviderInfo(
                        name=name,
                        kind=kind,  # type: ignore[arg-type]
                        label=provider.label,  # type: ignore[attr-defined]
                        available=availability.available,
                        reason=availability.reason,
                        experimental=availability.experimental,
                        requires=availability.requires,
                    )
                )

        mesh_availability = self._mesh.availability()
        infos.append(
            ProviderInfo(
                name=self._mesh.name,
                kind="mesh",
                label=self._mesh.label,
                available=mesh_availability.available,
                reason=mesh_availability.reason,
                requires=mesh_availability.requires,
            )
        )
        return infos

    #: Below this much total VRAM, only one heavyweight model stays resident.
    #: Stable Diffusion and a reconstruction model together leave a 6 GB card
    #: with a few hundred MB free, which is one step from an out-of-memory
    #: failure on the next request.
    SINGLE_MODEL_VRAM_BYTES = 12 * 1024**3

    def _vram_is_tight(self) -> bool:
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return False
            total = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            return False
        return total < self.SINGLE_MODEL_VRAM_BYTES

    def free_vram_for(self, keep) -> None:
        """Release every other model's GPU memory before ``keep`` runs.

        Only on cards too small to hold two models at once; a roomy GPU keeps
        both loaded so repeated runs stay fast.
        """
        if not self._vram_is_tight():
            return
        released = []
        for group in (self._image, self._threed):
            for name, provider in group.items():
                if provider is keep:
                    continue
                # Providers cache their weights under one of these names; only
                # release ones that actually hold something.
                loaded = getattr(provider, "_pipe", None) or getattr(provider, "_model", None)
                if loaded is not None:
                    provider.release()
                    released.append(name)
        if released:
            logger.info("Freed VRAM held by: %s", ", ".join(released))

    def release_all(self) -> None:
        """Free cached weights held by any provider."""
        for group in (self._image, self._threed):
            for provider in group.values():
                provider.release()  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def get_registry() -> ProviderRegistry:
    """Process-wide registry singleton."""
    return ProviderRegistry(get_settings())
