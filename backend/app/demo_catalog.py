"""The demo projects: the catalogue, and registering the committed ones.

The demo assets under ``storage/demo/<uuid>/`` are committed to the
repository, but project rows live in the SQLite database, which is not.
``scripts/seed_demo.py`` creates the rows while it generates the assets - and
that needs an image model - so a fresh clone used to open onto an empty
Gallery even though the files were sitting right there.

:func:`adopt_demo_projects` closes that gap. On startup it registers a row for
every catalogued demo whose files are on disk and whose row is missing, built
from those files alone: the specification from the prompt, the statistics
from the mesh, the views and exports from what is in the folder. It never
generates anything and never touches a row that already exists.

The catalogue and the discovery helpers import no database code, so
``scripts/export_static_site.py`` can use them with a minimal install.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.schemas import ExportFormat, ViewName

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Demo:
    slug: str
    name: str
    prompt: str
    seed: int
    #: Deterministic id so re-seeding replaces rather than duplicates.
    uuid: str
    #: 3D backend. Vehicles and organic shapes need the learned model; a
    #: silhouette hull turns them into a slab because diffusion models return
    #: three-quarter hero shots rather than the elevations carving requires.
    threed_provider: str | None = None


DEMOS: tuple[Demo, ...] = (
    Demo(
        slug="car",
        name="Futuristic Concept Car",
        prompt=(
            "A futuristic aerodynamic electric sports car, smooth silver body panels, "
            "light grey, 4.2 m long, 1.9 m wide, 1.2 m tall"
        ),
        seed=1201,
        uuid="d0000000-0000-4000-8000-000000000001",
        threed_provider="triposr",
    ),
    Demo(
        slug="wheelchair",
        name="Assistive Wheelchair Tray",
        prompt=(
            "A lightweight ergonomic wheelchair tray attachment in matte black ABS, "
            "45 cm x 30 cm x 3 cm, 3D printable, waterproof, for outdoor use"
        ),
        seed=2202,
        uuid="d0000000-0000-4000-8000-000000000002",
    ),
    Demo(
        slug="chair",
        name="Modern Oak Chair",
        prompt=(
            "A modern minimalist oak dining chair with a solid seat and tapered legs, "
            "45 cm x 45 cm x 90 cm, CNC machined"
        ),
        seed=3303,
        uuid="d0000000-0000-4000-8000-000000000003",
    ),
    Demo(
        slug="wearable",
        name="Smart Wearable Band",
        prompt=(
            "A smart wearable fitness band with a rounded silicone strap and a compact "
            "display module, light grey, 250 mm x 25 mm x 12 mm, injection moulded"
        ),
        seed=4404,
        uuid="d0000000-0000-4000-8000-000000000004",
        threed_provider="triposr",
    ),
)

#: The views were rendered by the local diffusers provider (sd-turbo), per
#: storage/demo/README.md. Per-view seeds were not recorded on disk, so an
#: adopted image carries no seed rather than a guessed one.
DEMO_IMAGE_PROVIDER = "diffusers"

#: Export suffixes a demo folder may hold, besides the ExportFormat values.
_EXTRA_EXPORT_SUFFIXES = ("stp",)


# --------------------------------------------------------------------------
# Discovery - what a demo folder actually contains
# --------------------------------------------------------------------------


def demo_model_path(directory: Path) -> Path | None:
    """The model the viewer should load: the processed mesh when there is one."""
    for name in ("model_processed.glb", "model.glb"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def demo_image_files(directory: Path) -> list[tuple[ViewName, Path]]:
    """``(view, path)`` for every ``images/<view>.png``, in canonical view order."""
    images = directory / "images"
    return [
        (view, images / f"{view.value}.png")
        for view in ViewName
        if (images / f"{view.value}.png").is_file()
    ]


def demo_export_files(directory: Path, project_id: str) -> dict[str, Path]:
    """``{format: path}`` for every ``exports/<id>.<format>`` present on disk."""
    exports = directory / "exports"
    formats = [fmt.value for fmt in ExportFormat] + list(_EXTRA_EXPORT_SUFFIXES)
    return {
        fmt: exports / f"{project_id}.{fmt}"
        for fmt in formats
        if (exports / f"{project_id}.{fmt}").is_file()
    }


def image_size(path: Path) -> tuple[int, int]:
    """Pixel size of an image, read from its header only."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as opened:
        return opened.size


# --------------------------------------------------------------------------
# Adoption - register rows for committed assets
# --------------------------------------------------------------------------


def adopt_demo_projects(demo_dir: Path | None = None,
                        demos: Iterable[Demo] = DEMOS) -> list[str]:
    """Register a project row for each demo that has files but no row.

    Idempotent, and a no-op when the demo folder is absent. Returns the ids it
    registered. A demo whose files cannot be read is skipped with a warning;
    it never stops the others, or the application, from starting.
    """
    from app.config import get_settings  # noqa: PLC0415
    from app.db import ProjectRow, session_scope, utcnow  # noqa: PLC0415
    from app.pipeline.prompt_parser import parse_prompt  # noqa: PLC0415
    from app.providers.mesh.trimesh_processor import (  # noqa: PLC0415
        load_mesh,
        mesh_statistics,
    )
    from app.schemas import GeneratedImage, ProjectStatus  # noqa: PLC0415
    from app.storage import relative_url, repository  # noqa: PLC0415

    root = demo_dir or get_settings().demo_dir
    if root is None or not root.is_dir():
        return []

    adopted: list[str] = []
    for demo in demos:
        directory = root / demo.uuid
        model_path = demo_model_path(directory)
        if model_path is None:
            continue
        with session_scope() as session:
            if session.get(ProjectRow, demo.uuid) is not None:
                continue

        created = False
        try:
            # Read everything first, so a bad file leaves no half-made row.
            stats = mesh_statistics(load_mesh(model_path), model_path)
            spec = parse_prompt(demo.prompt)
            now = utcnow()
            images = []
            for view, path in demo_image_files(directory):
                width, height = image_size(path)
                images.append(GeneratedImage(
                    view=view, path=str(path), url=relative_url(path) or "",
                    width=width, height=height, provider=DEMO_IMAGE_PROVIDER,
                    seed=None, created_at=now,
                ))
            exports = {fmt: str(path)
                       for fmt, path in demo_export_files(directory, demo.uuid).items()}
            processed = model_path.name == "model_processed.glb"

            repository.create(demo.prompt, name=demo.name, design_spec=spec,
                              is_demo=True, project_id=demo.uuid)
            created = True
            if images:
                repository.set_images(demo.uuid, images)
            repository.update(
                demo.uuid,
                model_path=str(model_path),
                model_format="glb",
                stats=stats,
                exports=exports,
                status=(ProjectStatus.MESH_PROCESSED if processed
                        else ProjectStatus.MODEL_READY).value,
                extra={"demo": {"slug": demo.slug, "registered_from": "committed demo assets"}},
            )
            repository.append_history(
                demo.uuid, "demo_registered",
                "Registered from the committed demo assets in storage/demo",
                {"model": model_path.name, "views": [image.view.value for image in images],
                 "exports": sorted(exports)},
            )
        except Exception:
            logger.warning("Could not register demo '%s' from %s", demo.slug, directory,
                           exc_info=True)
            if created:
                # Drop the row only - repository.delete would remove the
                # committed files along with it.
                with session_scope() as session:
                    row = session.get(ProjectRow, demo.uuid)
                    if row is not None:
                        session.delete(row)
            continue
        adopted.append(demo.uuid)

    if adopted:
        logger.info("Registered %d demo project(s) from %s", len(adopted), root)
    return adopted
