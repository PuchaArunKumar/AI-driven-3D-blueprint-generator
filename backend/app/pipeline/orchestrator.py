"""Pipeline orchestration.

Ties the providers together into the documented workflow:

    Text -> Design spec -> Multi-view images -> 3D -> Mesh -> CAD -> Export

Each entry point here is a *job worker*: it takes a :class:`~app.jobs.JobContext`,
reports real progress as it goes, and persists its output to the project before
returning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db import utcnow
from app.errors import ExportError, GenerationError, NotFoundError, ValidationError
from app.jobs import JobContext
from app.pipeline.prompt_parser import parse_prompt
from app.providers.base import ImageResult
from app.providers.mesh.trimesh_processor import convert, load_mesh, mesh_statistics
from app.providers.registry import get_registry
from app.schemas import (
    DesignSpec,
    ExportFormat,
    GeneratedImage,
    ProjectStatus,
    ViewName,
)
from app.storage import (
    exports_dir,
    images_dir,
    project_dir,
    raw_exports,
    relative_url,
    repository,
)

logger = logging.getLogger(__name__)

#: Formats that Blender must produce (trimesh cannot write them).
BLENDER_ONLY = {"blend"}
#: Formats that FreeCAD must produce.
FREECAD_ONLY = {"step", "stp"}


def _project_or_404(project_id: str):
    project = repository.get(project_id)
    if project is None:  # pragma: no cover - repository raises instead
        raise NotFoundError("That project no longer exists.")
    return project


def _images_from_project(project) -> list[ImageResult]:
    """Rebuild provider-level image records from what is stored on the project."""
    results: list[ImageResult] = []
    for image in project.images:
        path = Path(image.path)
        if path.exists():
            results.append(
                ImageResult(
                    view=image.view,
                    path=path,
                    width=image.width,
                    height=image.height,
                    provider=image.provider,
                    seed=image.seed,
                )
            )
    return results


# --------------------------------------------------------------------------
# Stage 1-2: prompt -> specification
# --------------------------------------------------------------------------


def analyse_prompt(project_id: str, prompt: str) -> DesignSpec:
    """Parse ``prompt`` into a design specification and store it."""
    spec = parse_prompt(prompt)
    repository.update(project_id, design_spec=spec, status=ProjectStatus.SPECIFIED.value)
    repository.append_history(project_id, "prompt_analysed",
                              f"Interpreted as: {spec.object}")
    return spec


# --------------------------------------------------------------------------
# Stage 3: multi-view images
# --------------------------------------------------------------------------


def generate_images(
    context: JobContext,
    project_id: str,
    views: list[ViewName],
    provider_name: str | None = None,
    seed: int | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    """Render the requested views and attach them to the project."""
    settings = get_settings()
    registry = get_registry()
    project = _project_or_404(project_id)

    context.start_stage("prompt_analysis")
    spec = project.design_spec or analyse_prompt(project_id, project.prompt)
    context.finish_stage("prompt_analysis", f"Object: {spec.object}")

    context.start_stage("design_spec")
    context.finish_stage("design_spec", f"{len(spec.materials)} material(s) identified")

    provider = registry.image_provider(provider_name)
    availability = provider.availability()
    if not availability.available:
        context.fail_stage("images", availability.reason)
        raise GenerationError(
            availability.reason,
            hint="Open Settings to choose a different image provider, or browse the "
                 "demo projects, which need no provider at all.",
        )

    registry.free_vram_for(provider)
    context.start_stage("images", f"Using {provider.label}")
    target_dir = images_dir(project_id, demo=project.is_demo)
    target_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results = provider.generate(
        spec,
        views,
        target_dir,
        seed=seed,
        width=settings.image_width,
        height=settings.image_height,
        progress=context.stage_progress("images"),
    )
    context.check_cancelled()

    kept: list[GeneratedImage] = []
    if not replace:
        # Preserve views the user did not ask to regenerate.
        regenerated = {result.view for result in results}
        kept = [image for image in project.images if image.view not in regenerated]

    now = utcnow()
    generated = [
        GeneratedImage(
            view=result.view,
            path=str(result.path),
            url=relative_url(result.path) or "",
            width=result.width,
            height=result.height,
            provider=result.provider,
            seed=result.seed,
            created_at=now,
        )
        for result in results
    ]
    ordered = sorted(kept + generated, key=lambda image: list(ViewName).index(image.view))
    repository.set_images(project_id, ordered)
    repository.update(project_id, status=ProjectStatus.IMAGES_READY.value)
    repository.append_history(
        project_id, "images_generated",
        f"{len(generated)} view(s) via {provider.label}",
        {"views": [image.view.value for image in generated],
         "seconds": round(time.time() - started, 2)},
    )
    context.finish_stage("images", f"{len(generated)} view(s) rendered")
    return {
        "images": [image.model_dump(mode="json") for image in ordered],
        "provider": provider.name,
    }


# --------------------------------------------------------------------------
# Stage 4: image -> 3D
# --------------------------------------------------------------------------


def generate_3d(
    context: JobContext,
    project_id: str,
    provider_name: str | None = None,
    resolution: int | None = None,
) -> dict[str, Any]:
    """Reconstruct a 3D model from the project's generated views."""
    settings = get_settings()
    registry = get_registry()
    project = _project_or_404(project_id)

    images = _images_from_project(project)
    if not images:
        raise ValidationError(
            "Generate the view images before running 3D reconstruction.",
            hint="Use 'Generate Views' first, or upload a reference image.",
        )

    provider = registry.threed_provider(provider_name)
    availability = provider.availability()
    if not availability.available:
        context.fail_stage("reconstruction", availability.reason)
        raise GenerationError(
            availability.reason,
            hint="Switch the 3D provider in Settings, or configure the endpoint "
                 "credentials it needs.",
        )

    registry.free_vram_for(provider)
    context.start_stage("reconstruction", f"Using {provider.label}")
    target_dir = project_dir(project_id, demo=project.is_demo)
    target_dir.mkdir(parents=True, exist_ok=True)

    result = provider.generate(
        images,
        target_dir,
        spec=project.design_spec,
        resolution=resolution or settings.voxel_resolution,
        progress=context.stage_progress("reconstruction"),
    )
    context.check_cancelled()

    mesh = load_mesh(result.path)
    stats = mesh_statistics(mesh, result.path, generation_time_s=result.generation_time_s)
    metadata = {**(project.project_metadata or {}), "reconstruction": result.metadata}

    repository.update(
        project_id,
        model_path=str(result.path),
        model_format=result.file_format,
        stats=stats,
        status=ProjectStatus.MODEL_READY.value,
        extra=metadata,
        exports={"glb": str(result.path)},
    )
    repository.append_history(
        project_id, "model_generated",
        f"{stats.triangles:,} triangles via {provider.label}",
        result.metadata,
    )
    context.finish_stage("reconstruction", f"{stats.triangles:,} triangles")
    return {
        "model_url": relative_url(result.path),
        "stats": stats.model_dump(mode="json"),
        "metadata": result.metadata,
    }


# --------------------------------------------------------------------------
# Stage 5: mesh processing
# --------------------------------------------------------------------------


def process_mesh(
    context: JobContext,
    project_id: str,
    *,
    smooth_iterations: int = 2,
    target_faces: int | None = None,
    fill_holes: bool = True,
    remove_duplicates: bool = True,
    recompute_normals: bool = True,
) -> dict[str, Any]:
    """Clean the generated mesh and replace the project's active model."""
    registry = get_registry()
    project = _project_or_404(project_id)
    if not project.model_path or not Path(project.model_path).exists():
        raise ValidationError(
            "Generate the 3D model before processing the mesh.",
            hint="Run 'Generate 3D' first.",
        )

    context.start_stage("mesh")
    processor = registry.mesh_processor()
    result = processor.process(
        Path(project.model_path),
        project_dir(project_id, demo=project.is_demo),
        smooth_iterations=smooth_iterations,
        target_faces=target_faces,
        fill_holes=fill_holes,
        remove_duplicates=remove_duplicates,
        recompute_normals=recompute_normals,
        progress=context.stage_progress("mesh"),
    )
    context.check_cancelled()

    mesh = load_mesh(result.path)
    stats = mesh_statistics(mesh, result.path)
    metadata = {**(project.project_metadata or {}), "mesh_processing": result.metadata}

    repository.update(
        project_id,
        model_path=str(result.path),
        model_format=result.file_format,
        stats=stats,
        status=ProjectStatus.MESH_PROCESSED.value,
        extra=metadata,
        # Previous exports were made from the old mesh, so drop them.
        exports={"glb": str(result.path)},
    )
    repository.append_history(
        project_id, "mesh_processed",
        "; ".join(result.metadata.get("operations", [])) or "cleaned",
        result.metadata,
    )
    context.finish_stage("mesh", f"{stats.triangles:,} triangles")
    return {
        "model_url": relative_url(result.path),
        "stats": stats.model_dump(mode="json"),
        "metadata": result.metadata,
    }


# --------------------------------------------------------------------------
# Stage 6-7: CAD conversion and export
# --------------------------------------------------------------------------


def export_model(
    context: JobContext,
    project_id: str,
    formats: list[ExportFormat],
) -> dict[str, Any]:
    """Write the model in each requested format, skipping ones that fail."""
    registry = get_registry()
    project = _project_or_404(project_id)
    if not project.model_path or not Path(project.model_path).exists():
        raise ValidationError(
            "There is no model to export yet.",
            hint="Generate the 3D model first.",
        )

    source = Path(project.model_path)
    target_dir = exports_dir(project_id, demo=project.is_demo)
    target_dir.mkdir(parents=True, exist_ok=True)
    available, unavailable = registry.available_export_formats()

    produced: dict[str, str] = {}
    failures: dict[str, str] = {}
    needs_cad = any(fmt.value in BLENDER_ONLY | FREECAD_ONLY for fmt in formats)

    if needs_cad:
        context.start_stage("cad")
    context.start_stage("export")

    for index, fmt in enumerate(formats):
        context.check_cancelled()
        name = fmt.value
        context.progress(index / max(1, len(formats)), f"Exporting {name.upper()}")

        if name not in available:
            failures[name] = unavailable.get(name, "This format is not available.")
            continue

        output_path = target_dir / f"{project_id}.{name}"
        try:
            if name in BLENDER_ONLY:
                registry.blender.export(source, output_path, name)
            elif name in FREECAD_ONLY:
                registry.freecad.export(source, output_path, name)
            else:
                convert(source, output_path)
            produced[name] = str(output_path)
        except Exception as exc:
            reason = getattr(exc, "user_message", None) or "Conversion failed."
            failures[name] = reason
            logger.warning("Export of %s failed for %s: %s", name, project_id, exc,
                           exc_info=True)

    if needs_cad:
        cad_formats = [name for name in produced if name in BLENDER_ONLY | FREECAD_ONLY]
        if cad_formats:
            context.finish_stage("cad", ", ".join(sorted(cad_formats)).upper())
        else:
            context.skip_stage("cad", "No CAD format was produced")

    if not produced:
        context.fail_stage("export", "; ".join(failures.values())[:200])
        raise ExportError(
            "None of the requested formats could be exported.",
            hint="; ".join(f"{name.upper()}: {reason}" for name, reason in failures.items()),
        )

    merged = {**raw_exports(project_id), **produced}
    repository.update(project_id, exports=merged)
    repository.append_history(
        project_id, "exported", ", ".join(sorted(produced)).upper(),
        {"failures": failures} if failures else None,
    )
    context.finish_stage("export", ", ".join(sorted(produced)).upper())

    return {
        "exports": {name: relative_url(path) for name, path in merged.items()},
        "produced": sorted(produced),
        "failed": failures,
    }


# --------------------------------------------------------------------------
# Blueprint data
# --------------------------------------------------------------------------


def blueprint_data(project_id: str) -> dict[str, Any]:
    """Technical line drawings projected from the real geometry."""
    from app.pipeline.blueprint import technical_drawing  # noqa: PLC0415

    project = _project_or_404(project_id)
    if not project.model_path or not Path(project.model_path).exists():
        raise ValidationError("Generate the 3D model before opening the blueprint view.")

    mesh = load_mesh(Path(project.model_path))
    stats = mesh_statistics(mesh, Path(project.model_path))
    return {
        "views": technical_drawing(mesh),
        "stats": stats.model_dump(mode="json"),
        "spec": project.design_spec.model_dump(mode="json") if project.design_spec else None,
        "units": "m",
    }


def blueprint_sheet(project_id: str, theme: str = "white") -> str:
    """Render the combined third-angle blueprint sheet as SVG."""
    from app.pipeline.blueprint_sheet import render_blueprint_sheet  # noqa: PLC0415

    data = blueprint_data(project_id)
    project = _project_or_404(project_id)
    return render_blueprint_sheet(
        data["views"], data["stats"], data["spec"],
        name=project.name, theme_name=theme,
    )


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------


def run_full_pipeline(
    context: JobContext,
    project_id: str,
    views: list[ViewName],
    *,
    image_provider: str | None = None,
    threed_provider: str | None = None,
    seed: int | None = None,
    resolution: int | None = None,
    export_formats: list[ExportFormat] | None = None,
) -> dict[str, Any]:
    """Run every stage end to end, as the Studio's single 'Generate' button does."""
    result: dict[str, Any] = {}
    result["images"] = generate_images(context, project_id, views, image_provider, seed)
    result["model"] = generate_3d(context, project_id, threed_provider, resolution)
    result["mesh"] = process_mesh(context, project_id)
    result["export"] = export_model(
        context, project_id, export_formats or [ExportFormat.GLB, ExportFormat.STL]
    )
    return result
