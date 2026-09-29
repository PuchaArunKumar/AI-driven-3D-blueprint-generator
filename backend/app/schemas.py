"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class ViewName(str, Enum):
    """Canonical camera positions used for multi-view generation."""

    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    PERSPECTIVE = "perspective"


DEFAULT_VIEWS: list[ViewName] = [
    ViewName.FRONT,
    ViewName.REAR,
    ViewName.LEFT,
    ViewName.RIGHT,
    ViewName.TOP,
    ViewName.PERSPECTIVE,
]


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProjectStatus(str, Enum):
    DRAFT = "draft"
    SPECIFIED = "specified"
    IMAGES_READY = "images_ready"
    MODEL_READY = "model_ready"
    MESH_PROCESSED = "mesh_processed"
    FAILED = "failed"


class ExportFormat(str, Enum):
    GLB = "glb"
    GLTF = "gltf"
    OBJ = "obj"
    PLY = "ply"
    STL = "stl"
    BLEND = "blend"
    STEP = "step"


# --------------------------------------------------------------------------
# Design specification
# --------------------------------------------------------------------------


class Dimensions(BaseModel):
    """Overall bounding dimensions of the designed product."""

    length_mm: float | None = Field(default=None, gt=0, le=100_000)
    width_mm: float | None = Field(default=None, gt=0, le=100_000)
    height_mm: float | None = Field(default=None, gt=0, le=100_000)
    weight_g: float | None = Field(default=None, gt=0, le=10_000_000)

    def described(self) -> str:
        """Human-readable dimension phrase used when building image prompts."""
        parts: list[str] = []
        if self.length_mm:
            parts.append(f"{self.length_mm:g} mm long")
        if self.width_mm:
            parts.append(f"{self.width_mm:g} mm wide")
        if self.height_mm:
            parts.append(f"{self.height_mm:g} mm tall")
        return ", ".join(parts)


class DesignSpec(BaseModel):
    """Structured interpretation of a natural-language design prompt.

    Produced by :mod:`app.pipeline.prompt_parser` and editable by the user
    before any image is generated.
    """

    object: str = Field(..., min_length=1, max_length=200)
    purpose: str = Field(default="", max_length=1000)
    dimensions: Dimensions = Field(default_factory=Dimensions)
    materials: list[str] = Field(default_factory=list)
    style: str = Field(default="", max_length=200)
    color: str = Field(default="", max_length=200)
    constraints: list[str] = Field(default_factory=list)
    manufacturing_requirements: list[str] = Field(default_factory=list)
    accessibility_requirements: list[str] = Field(default_factory=list)
    intended_use: str = Field(default="", max_length=500)
    tolerances_mm: float | None = Field(default=None, gt=0, le=100)

    @field_validator(
        "materials",
        "constraints",
        "manufacturing_requirements",
        "accessibility_requirements",
        mode="before",
    )
    @classmethod
    def _clean_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()][:20]


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------


class GeneratedImage(BaseModel):
    view: ViewName
    path: str
    url: str
    width: int
    height: int
    provider: str
    seed: int | None = None
    created_at: datetime
    is_reference: bool = False


class ModelStats(BaseModel):
    """Geometry statistics computed from the actual mesh."""

    vertices: int
    faces: int
    triangles: int
    bbox_min: list[float]
    bbox_max: list[float]
    width: float
    height: float
    depth: float
    volume: float | None = None
    surface_area: float | None = None
    is_watertight: bool = False
    file_size_bytes: int = 0
    file_format: str = ""
    generation_time_s: float | None = None


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=3, max_length=2000)
    name: str | None = Field(default=None, max_length=200)

    @field_validator("prompt")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("prompt must contain at least 3 non-whitespace characters")
        return cleaned


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=200)
    design_spec: DesignSpec | None = None


class ProjectSummary(BaseModel):
    id: str
    name: str
    prompt: str
    status: ProjectStatus
    thumbnail_url: str | None = None
    model_format: str | None = None
    is_demo: bool = False
    created_at: datetime
    updated_at: datetime


class Project(ProjectSummary):
    design_spec: DesignSpec | None = None
    images: list[GeneratedImage] = Field(default_factory=list)
    model_url: str | None = None
    model_path: str | None = None
    stats: ModelStats | None = None
    exports: dict[str, str] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    project_metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata")

    model_config = ConfigDict(populate_by_name=True)


# --------------------------------------------------------------------------
# Generation requests
# --------------------------------------------------------------------------


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    views: list[ViewName] = Field(default_factory=lambda: list(DEFAULT_VIEWS))
    provider: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    replace: bool = True

    @field_validator("views")
    @classmethod
    def _at_least_one(cls, value: list[ViewName]) -> list[ViewName]:
        if not value:
            raise ValueError("at least one view must be requested")
        seen: set[ViewName] = set()
        ordered: list[ViewName] = []
        for view in value:
            if view not in seen:
                seen.add(view)
                ordered.append(view)
        return ordered


class ThreeDGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    provider: str | None = None
    resolution: int | None = Field(default=None, ge=32, le=384)
    #: Used by text-conditioned providers only; random (and recorded) when unset.
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class MeshProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    smooth_iterations: int = Field(default=2, ge=0, le=20)
    target_faces: int | None = Field(default=None, ge=100, le=2_000_000)
    fill_holes: bool = True
    remove_duplicates: bool = True
    recompute_normals: bool = True


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formats: list[ExportFormat] = Field(default_factory=lambda: [ExportFormat.GLB])

    @field_validator("formats")
    @classmethod
    def _at_least_one(cls, value: list[ExportFormat]) -> list[ExportFormat]:
        if not value:
            raise ValueError("at least one export format must be requested")
        return value


class SpecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=3, max_length=2000)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class JobStage(BaseModel):
    key: str
    label: str
    status: StageStatus = StageStatus.PENDING
    detail: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Job(BaseModel):
    id: str
    project_id: str | None = None
    kind: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    hint: str = ""
    stages: list[JobStage] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------
# System / settings
# --------------------------------------------------------------------------


class ProviderInfo(BaseModel):
    name: str
    kind: Literal["image", "threed", "mesh", "cad"]
    label: str
    available: bool
    reason: str = ""
    experimental: bool = False
    requires: list[str] = Field(default_factory=list)


class HardwareInfo(BaseModel):
    cpu: str
    cpu_cores: int
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None
    gpu: str | None = None
    vram_total_gb: float | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    torch_version: str | None = None
    platform: str = ""


class CapabilityInfo(BaseModel):
    """What the running installation can actually do, right now."""

    export_formats: list[str]
    unavailable_export_formats: dict[str, str] = Field(default_factory=dict)
    blender: bool = False
    blender_version: str | None = None
    freecad: bool = False
    diffusers_installed: bool = False


class SystemStatus(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    hardware: HardwareInfo
    capabilities: CapabilityInfo
    providers: list[ProviderInfo]
    settings: dict[str, Any]
    project_count: int
    active_jobs: int
