"""Export the demo projects as a static, read-only API snapshot.

The GitHub Pages build has no backend. Its ``api.ts`` reads this snapshot
instead, so the Gallery, Project Details, 3D viewer, blueprint view and
downloads work for the committed demos with nothing running server-side.

Everything is computed from the files in ``storage/demo`` with the same code
the backend serves them with: the specification from the prompt
(:func:`parse_prompt`), statistics from the mesh (:func:`mesh_statistics`),
the technical drawing from :func:`technical_drawing` and both sheets from
:func:`render_blueprint_sheet`. Nothing is generated or invented.

Minimal dependencies - no torch, diffusers, rembg, FastAPI or SQLAlchemy::

    pip install "numpy>=1.26" "scipy>=1.13" "scikit-image>=0.24" "trimesh>=4.5" "fast-simplification>=0.1.7" "Pillow>=10.4" "pydantic>=2.9,<3.0" "pydantic-settings>=2.6,<3.0"

(fast-simplification decimates the mesh before the drawing exactly as the
backend does; without it the drawing is traced from the full mesh.)

Usage (from the repository root)::

    python backend/scripts/export_static_site.py --out frontend/public/static-api

Layout written under ``--out``::

    projects.json                        ProjectSummary[] (demos only)
    projects/<id>.json                   Project
    projects/<id>/blueprint.json         BlueprintData
    projects/<id>/blueprint-white.svg    combined sheet, white
    projects/<id>/blueprint-blueprint.svg
    files/demo/<id>/...                  the images, model and exports the JSON references

Every URL inside the JSON is relative to the site root with no leading slash
(``static-api/files/demo/<id>/model_processed.glb``); the frontend anchors it
to its base path. Prompt parsing is rules-only here (``SEMANTIC_PROMPT=false``),
so the specification does not depend on whether a transformer model happens to
be installed. Timestamps are the asset files' modification times.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Deterministic, dependency-light prompt parsing. Set before app.config loads.
os.environ.setdefault("SEMANTIC_PROMPT", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.demo_catalog import (  # noqa: E402
    DEMO_IMAGE_PROVIDER,
    DEMOS,
    Demo,
    demo_export_files,
    demo_image_files,
    demo_model_path,
    image_size,
)
from app.pipeline.blueprint import technical_drawing  # noqa: E402
from app.pipeline.blueprint_sheet import render_blueprint_sheet  # noqa: E402
from app.pipeline.prompt_parser import parse_prompt  # noqa: E402
from app.providers.mesh.trimesh_processor import (  # noqa: E402
    load_mesh,
    mesh_statistics,
)
from app.schemas import (  # noqa: E402
    GeneratedImage,
    Project,
    ProjectStatus,
    ProjectSummary,
)

logger = logging.getLogger("export_static_site")

#: Written into the output directory so a re-export may safely replace it.
MARKER = ".static-api"
DEFAULT_URL_PREFIX = "static-api/"
THEMES = ("white", "blueprint")


class Snapshot:
    """Collects what was written, for the closing summary."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.written: list[tuple[Path, int]] = []

    def write_json(self, relative: str, payload, *, compact: bool = False) -> None:
        if compact:
            text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False)
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        self.write_text(relative, text + "\n")

    def write_text(self, relative: str, text: str) -> None:
        target = self.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.written.append((target, target.stat().st_size))

    def copy(self, source: Path, relative: str) -> None:
        target = self.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.written.append((target, target.stat().st_size))


def _timestamp(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _repo_relative(path: Path) -> str:
    """A path for display that never leaks the exporting machine's layout."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def export_demo(demo: Demo, directory: Path, snapshot: Snapshot, prefix: str) -> Project | None:
    """Write one demo's JSON, sheets and files. ``None`` when it has no model."""
    model_path = demo_model_path(directory)
    if model_path is None:
        logger.warning("Demo '%s': no model in %s - skipped", demo.slug, directory)
        return None

    started = time.time()
    files_root = f"files/demo/{demo.uuid}"

    def url(relative_file: str) -> str:
        return f"{prefix}{files_root}/{relative_file}"

    spec = parse_prompt(demo.prompt)
    mesh = load_mesh(model_path)
    stats = mesh_statistics(mesh, model_path)
    stamp = _timestamp(model_path)

    images: list[GeneratedImage] = []
    for view, path in demo_image_files(directory):
        relative_file = f"images/{path.name}"
        snapshot.copy(path, f"{files_root}/{relative_file}")
        width, height = image_size(path)
        images.append(GeneratedImage(
            view=view, path=_repo_relative(path), url=url(relative_file),
            width=width, height=height, provider=DEMO_IMAGE_PROVIDER, seed=None,
            created_at=_timestamp(path),
        ))

    snapshot.copy(model_path, f"{files_root}/{model_path.name}")

    exports: dict[str, str] = {}
    for fmt, path in demo_export_files(directory, demo.uuid).items():
        relative_file = f"exports/{path.name}"
        snapshot.copy(path, f"{files_root}/{relative_file}")
        exports[fmt] = url(relative_file)

    processed = model_path.name == "model_processed.glb"
    project = Project(
        id=demo.uuid,
        name=demo.name,
        prompt=demo.prompt,
        status=ProjectStatus.MESH_PROCESSED if processed else ProjectStatus.MODEL_READY,
        thumbnail_url=images[0].url if images else None,
        model_format="glb",
        is_demo=True,
        created_at=stamp,
        updated_at=stamp,
        design_spec=spec,
        images=images,
        model_url=url(model_path.name),
        model_path=_repo_relative(model_path),
        stats=stats,
        exports=exports,
        history=[{
            "event": "static_snapshot",
            "detail": "Exported for the hosted demo from the committed assets in storage/demo",
            "at": stamp.isoformat(),
        }],
        metadata={"demo": {"slug": demo.slug, "static_snapshot": True}},
    )
    snapshot.write_json(f"projects/{demo.uuid}.json",
                        project.model_dump(mode="json", by_alias=True))

    # Exactly what GET /api/projects/<id>/blueprint and blueprint.svg return.
    views = technical_drawing(mesh)
    stats_json = stats.model_dump(mode="json")
    spec_json = spec.model_dump(mode="json")
    # Compact: thousands of polyline points, fetched by the browser.
    snapshot.write_json(f"projects/{demo.uuid}/blueprint.json", {
        "views": views, "stats": stats_json, "spec": spec_json, "units": "m",
    }, compact=True)
    for theme in THEMES:
        snapshot.write_text(
            f"projects/{demo.uuid}/blueprint-{theme}.svg",
            render_blueprint_sheet(views, stats_json, spec_json, name=demo.name,
                                   theme_name=theme),
        )

    logger.info("Demo '%s': %s triangles, %d view(s), exports %s (%.1fs)", demo.slug,
                f"{stats.triangles:,}", len(images), ", ".join(sorted(exports)) or "none",
                time.time() - started)
    return project


def prepare_output(out: Path, force: bool) -> None:
    """Create ``out``, clearing a previous snapshot but never anything else."""
    if out.exists() and any(out.iterdir()):
        if not (out / MARKER).exists() and not force:
            raise SystemExit(
                f"{out} is not empty and holds no previous snapshot ({MARKER}). "
                "Choose an empty directory, or pass --force to write into it anyway."
            )
        for name in ("projects.json", "projects", "files"):
            target = out / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(
        "Generated by backend/scripts/export_static_site.py - safe to delete and regenerate.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the demos as a static API snapshot.")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, e.g. frontend/public/static-api")
    parser.add_argument("--demo-dir", type=Path, default=None,
                        help="demo assets (default: the configured storage/demo)")
    parser.add_argument("--url-prefix", default=DEFAULT_URL_PREFIX,
                        help=f"prefix of every URL in the JSON (default '{DEFAULT_URL_PREFIX}')")
    parser.add_argument("--force", action="store_true",
                        help="write into a non-empty directory that holds no previous snapshot")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    # The drawing code logs every view at INFO; the summary below is enough.
    logging.getLogger("app").setLevel(logging.WARNING)

    demo_dir = (arguments.demo_dir or get_settings().demo_dir).resolve()
    out = arguments.out.resolve()
    prefix = arguments.url_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    if prefix.startswith("/"):
        logger.error("--url-prefix must be relative to the site root (no leading slash)")
        return 2
    if not demo_dir.is_dir():
        logger.error("Demo directory not found: %s", demo_dir)
        return 1

    prepare_output(out, arguments.force)
    snapshot = Snapshot(out)
    started = time.time()

    projects: list[Project] = []
    for demo in DEMOS:
        project = export_demo(demo, demo_dir / demo.uuid, snapshot, prefix)
        if project is not None:
            projects.append(project)

    if not projects:
        logger.error("No demo assets found under %s - nothing to export.", demo_dir)
        return 1

    summaries = [
        ProjectSummary.model_validate(project.model_dump(by_alias=True)).model_dump(mode="json")
        for project in projects
    ]
    snapshot.write_json("projects.json", summaries)

    total = sum(size for _, size in snapshot.written)
    print(f"\nStatic API snapshot written to {out}")
    for path, size in sorted(snapshot.written):
        print(f"  {path.relative_to(out).as_posix():<78} {size / 1024:>9.1f} KB")
    print(f"{len(projects)} demo project(s), {len(snapshot.written)} files, "
          f"{total / 1e6:.1f} MB, {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
