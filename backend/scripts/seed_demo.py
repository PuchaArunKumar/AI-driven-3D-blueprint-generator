"""Seed the demo projects.

Demo mode is not a mock: this script runs the *real* pipeline and stores the
resulting assets under ``storage/demo``.  Anyone can then open, rotate,
inspect and download those models with no provider configured and no API cost.

Usage (from the ``backend`` directory)::

    python scripts/seed_demo.py                 # generate any missing demos
    python scripts/seed_demo.py --force         # regenerate all of them
    python scripts/seed_demo.py --only car      # just one, by slug
    python scripts/seed_demo.py --list          # show what would run

Each demo is marked ``is_demo`` so the UI can label it "Demo Asset" and never
present it as something the visitor just generated.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

# Allow running as `python scripts/seed_demo.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db import ProjectRow, init_db, session_scope  # noqa: E402
from app.errors import BlueprintError  # noqa: E402
from app.jobs import JobContext, JobRecord, manager  # noqa: E402
from app.logging_conf import configure_logging  # noqa: E402
from app.pipeline import orchestrator  # noqa: E402
from app.pipeline.prompt_parser import parse_prompt  # noqa: E402
from app.providers.registry import get_registry  # noqa: E402
from app.schemas import ExportFormat, ViewName  # noqa: E402
from app.storage import repository  # noqa: E402

logger = logging.getLogger("seed_demo")

#: Demo meshes are decimated so the committed assets stay small and load
#: quickly in the browser. Carving at 128 voxels produces far more detail
#: than a hull actually carries.
DEMO_TARGET_FACES = 40_000

#: Formats committed with the demos. The rest are exported on demand from
#: the UI, which keeps the repository small and exercises the export path.
DEMO_FORMATS = (ExportFormat.GLB, ExportFormat.STL, ExportFormat.BLEND)

CARVING_VIEWS = [
    ViewName.FRONT,
    ViewName.REAR,
    ViewName.LEFT,
    ViewName.RIGHT,
    ViewName.TOP,
]


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


class ConsoleContext(JobContext):
    """A JobContext that logs to the console instead of a WebSocket."""

    def __init__(self, label: str) -> None:
        record = JobRecord(id=str(uuid.uuid4()), kind="seed", project_id=None)
        super().__init__(manager, record)
        self._label = label

    def progress(self, fraction: float, message: str = "") -> None:
        if message:
            logger.info("  [%s] %3d%%  %s", self._label, int(fraction * 100), message)

    def start_stage(self, key: str, detail: str = "") -> None:
        logger.info("  [%s] -> %s %s", self._label, key, detail)

    def finish_stage(self, key: str, detail: str = "") -> None:
        logger.info("  [%s] OK %s %s", self._label, key, detail)

    def skip_stage(self, key: str, detail: str = "") -> None:
        logger.info("  [%s] -- %s %s", self._label, key, detail)

    def fail_stage(self, key: str, detail: str = "") -> None:
        logger.error("  [%s] XX %s %s", self._label, key, detail)

    def check_cancelled(self) -> None:
        return

    def stage_progress(self, key: str):
        def report(fraction: float, message: str = "") -> None:
            self.progress(fraction, message)

        return report


def _exists(demo: Demo) -> bool:
    with session_scope() as session:
        row = session.get(ProjectRow, demo.uuid)
        return row is not None and bool(row.model_path) and Path(row.model_path).exists()


def seed_one(demo: Demo, export_formats: list[ExportFormat]) -> bool:
    """Run the full pipeline for one demo. Returns True on success."""
    logger.info("Seeding demo '%s' - %s", demo.slug, demo.name)
    context = ConsoleContext(demo.slug)

    # Replace any previous attempt so a retry starts clean.
    with contextlib.suppress(BlueprintError):
        repository.delete(demo.uuid)

    spec = parse_prompt(demo.prompt)
    repository.create(
        demo.prompt,
        name=demo.name,
        design_spec=spec,
        is_demo=True,
        project_id=demo.uuid,
    )

    try:
        orchestrator.generate_images(context, demo.uuid, CARVING_VIEWS, seed=demo.seed)
        orchestrator.generate_3d(context, demo.uuid, demo.threed_provider)
        orchestrator.process_mesh(context, demo.uuid, smooth_iterations=2,
                                  target_faces=DEMO_TARGET_FACES)
        orchestrator.export_model(context, demo.uuid, export_formats)
    except BlueprintError as exc:
        logger.error("Demo '%s' failed: %s", demo.slug, exc.user_message)
        if exc.hint:
            logger.error("  hint: %s", exc.hint)
        return False

    project = repository.get(demo.uuid)
    logger.info(
        "Demo '%s' ready: %s triangles, exports: %s",
        demo.slug,
        f"{project.stats.triangles:,}" if project.stats else "?",
        ", ".join(sorted(project.exports)) or "none",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the demo projects.")
    parser.add_argument("--force", action="store_true",
                        help="regenerate demos that already exist")
    parser.add_argument("--only", metavar="SLUG",
                        help="seed a single demo by slug")
    parser.add_argument("--list", action="store_true",
                        help="list the demos and exit")
    arguments = parser.parse_args()

    settings = get_settings()
    configure_logging("INFO", settings.storage_dir / "logs")
    settings.ensure_directories()
    init_db()

    selected = [demo for demo in DEMOS if not arguments.only or demo.slug == arguments.only]
    if arguments.only and not selected:
        logger.error("No demo with slug '%s'. Available: %s",
                     arguments.only, ", ".join(demo.slug for demo in DEMOS))
        return 2

    if arguments.list:
        for demo in selected:
            logger.info("%-12s %-28s %s", demo.slug, demo.name,
                        "present" if _exists(demo) else "missing")
        return 0

    registry = get_registry()
    image_availability = registry.image_provider().availability()
    if not image_availability.available:
        logger.error("Cannot seed demos: %s", image_availability.reason)
        logger.error("Configure an image provider first - see README 'AI provider configuration'.")
        return 1

    available_formats, _ = registry.available_export_formats()
    export_formats = [fmt for fmt in DEMO_FORMATS if fmt.value in available_formats]
    logger.info("Exporting demos as: %s", ", ".join(fmt.value for fmt in export_formats))

    failures = 0
    for demo in selected:
        if _exists(demo) and not arguments.force:
            logger.info("Demo '%s' already present - skipping (use --force to rebuild)",
                        demo.slug)
            continue
        if not seed_one(demo, export_formats):
            failures += 1

    registry.release_all()
    if failures:
        logger.error("%d demo(s) failed to seed", failures)
        return 1
    logger.info("All demos seeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
