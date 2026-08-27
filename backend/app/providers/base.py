"""Abstract provider interfaces.

Every stage of the pipeline is expressed as a small interface so that a
provider can be swapped without touching the orchestrator, the API or the UI.

Each provider reports its own availability through :meth:`Provider.availability`.
The application never assumes a provider works - it asks first, and surfaces
the reason to the user when it does not.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from app.schemas import DesignSpec, ViewName

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:  # pragma: no cover - trivial
    """Default progress sink used when a caller does not supply one."""


@dataclass(slots=True)
class Availability:
    """Whether a provider can run right now, and why not if it cannot."""

    available: bool
    reason: str = ""
    experimental: bool = False
    requires: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, *, experimental: bool = False) -> Availability:
        return cls(available=True, experimental=experimental)

    @classmethod
    def missing(cls, reason: str, requires: list[str] | None = None,
                *, experimental: bool = False) -> Availability:
        return cls(available=False, reason=reason,
                   requires=requires or [], experimental=experimental)


@dataclass(slots=True)
class ImageResult:
    """One rendered view produced by an :class:`ImageGenerator`."""

    view: ViewName
    path: Path
    width: int
    height: int
    provider: str
    seed: int | None = None


@dataclass(slots=True)
class ModelResult:
    """A generated 3D asset plus whatever provenance the provider can give."""

    path: Path
    file_format: str
    provider: str
    generation_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(abc.ABC):
    """Common behaviour shared by every provider."""

    name: str = "provider"
    label: str = "Provider"
    kind: Literal["image", "threed", "mesh", "cad"] = "image"

    @abc.abstractmethod
    def availability(self) -> Availability:
        """Report whether this provider can execute on this machine, now."""

    def release(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Free any cached weights or GPU memory. Safe to call repeatedly."""


class ImageGenerator(Provider):
    """Turns a design specification into one image per requested view."""

    kind = "image"

    @abc.abstractmethod
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
        """Render ``views`` of the product described by ``spec``."""


class ThreeDGenerator(Provider):
    """Turns a set of view images into a 3D asset."""

    kind = "threed"

    @abc.abstractmethod
    def generate(
        self,
        images: list[ImageResult],
        output_dir: Path,
        *,
        spec: DesignSpec | None = None,
        resolution: int = 128,
        progress: ProgressCallback = _noop_progress,
    ) -> ModelResult:
        """Reconstruct a 3D model from ``images``."""


class MeshProcessor(Provider):
    """Cleans and repairs a generated mesh."""

    kind = "mesh"

    @abc.abstractmethod
    def process(
        self,
        model_path: Path,
        output_dir: Path,
        *,
        smooth_iterations: int = 2,
        target_faces: int | None = None,
        fill_holes: bool = True,
        remove_duplicates: bool = True,
        recompute_normals: bool = True,
        progress: ProgressCallback = _noop_progress,
    ) -> ModelResult:
        """Return a cleaned copy of the mesh at ``model_path``."""


class CADExporter(Provider):
    """Converts a mesh into a CAD-tool-native or interchange format."""

    kind = "cad"

    #: Formats this exporter can produce when it is available.
    formats: tuple[str, ...] = ()

    @abc.abstractmethod
    def export(self, model_path: Path, output_path: Path, fmt: str) -> Path:
        """Write ``model_path`` to ``output_path`` in ``fmt``; return the path."""


class SupportsThumbnail(Protocol):
    """Optional capability: render a preview PNG for a model."""

    def render_thumbnail(self, model_path: Path, output_path: Path) -> Path: ...
