"""Generation, mesh-processing and export endpoints.

Every one of these submits a background job and returns immediately with the
job record, so the frontend is never blocked on a multi-minute model.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field

from app.jobs import PIPELINE_STAGES, JobContext, manager
from app.pipeline import orchestrator
from app.schemas import (
    DEFAULT_VIEWS,
    ExportFormat,
    ExportRequest,
    ImageGenerationRequest,
    Job,
    MeshProcessRequest,
    ThreeDGenerationRequest,
    ViewName,
)
from app.storage import repository, validate_project_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["generation"])


def _stages(*keys: str) -> list[tuple[str, str]]:
    """Pick the pipeline stages relevant to one job."""
    lookup = dict(PIPELINE_STAGES)
    return [(key, lookup[key]) for key in keys]


@router.post("/generate/images", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def generate_images(payload: ImageGenerationRequest) -> Job:
    """Generate (or regenerate) the requested views."""
    project_id = validate_project_id(payload.project_id)
    repository.get(project_id)  # 404s early if the project is gone

    def worker(context: JobContext) -> dict:
        return orchestrator.generate_images(
            context, project_id, payload.views, payload.provider,
            payload.seed, payload.replace,
        )

    return manager.submit(
        "images", worker, project_id=project_id,
        stages=_stages("prompt_analysis", "design_spec", "images"),
    )


@router.post("/generate/3d", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def generate_3d(payload: ThreeDGenerationRequest) -> Job:
    """Reconstruct a 3D model from the project's views (or its prompt, for a text provider)."""
    project_id = validate_project_id(payload.project_id)
    repository.get(project_id)

    def worker(context: JobContext) -> dict:
        return orchestrator.generate_3d(
            context, project_id, payload.provider, payload.resolution, seed=payload.seed
        )

    return manager.submit(
        "3d", worker, project_id=project_id, stages=_stages("reconstruction"),
    )


@router.post("/process/mesh", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def process_mesh(payload: MeshProcessRequest) -> Job:
    """Clean, repair and optionally decimate the generated mesh."""
    project_id = validate_project_id(payload.project_id)
    repository.get(project_id)

    def worker(context: JobContext) -> dict:
        return orchestrator.process_mesh(
            context, project_id,
            smooth_iterations=payload.smooth_iterations,
            target_faces=payload.target_faces,
            fill_holes=payload.fill_holes,
            remove_duplicates=payload.remove_duplicates,
            recompute_normals=payload.recompute_normals,
        )

    return manager.submit(
        "mesh", worker, project_id=project_id, stages=_stages("mesh"),
    )


@router.post("/export/{project_id}", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def export_project(project_id: str, payload: ExportRequest) -> Job:
    """Export the model in the requested formats."""
    project_id = validate_project_id(project_id)
    repository.get(project_id)
    formats = payload.formats

    def worker(context: JobContext) -> dict:
        return orchestrator.export_model(context, project_id, formats)

    needs_cad = any(fmt.value in {"blend", "step", "stp"} for fmt in formats)
    stage_keys = ("cad", "export") if needs_cad else ("export",)
    return manager.submit(
        "export", worker, project_id=project_id, stages=_stages(*stage_keys),
    )


class FullPipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    views: list[ViewName] = Field(default_factory=lambda: list(DEFAULT_VIEWS))
    image_provider: str | None = None
    threed_provider: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    resolution: int | None = Field(default=None, ge=32, le=384)
    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.GLB, ExportFormat.STL]
    )


@router.post("/generate/pipeline", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def run_pipeline(payload: FullPipelineRequest) -> Job:
    """Run every stage end to end - the Studio's single 'Generate Design' button."""
    project_id = validate_project_id(payload.project_id)
    repository.get(project_id)

    def worker(context: JobContext) -> dict:
        return orchestrator.run_full_pipeline(
            context, project_id, payload.views,
            image_provider=payload.image_provider,
            threed_provider=payload.threed_provider,
            seed=payload.seed,
            resolution=payload.resolution,
            export_formats=payload.export_formats,
        )

    return manager.submit(
        "pipeline", worker, project_id=project_id, stages=PIPELINE_STAGES,
    )
