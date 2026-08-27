"""Project CRUD, reference-image upload and blueprint data."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Query, Response, UploadFile, status

from app.config import get_settings
from app.db import utcnow
from app.errors import ValidationError
from app.pipeline.orchestrator import analyse_prompt, blueprint_data, blueprint_sheet
from app.pipeline.prompt_parser import parse_prompt
from app.schemas import (
    DesignSpec,
    GeneratedImage,
    Project,
    ProjectCreate,
    ProjectSummary,
    ProjectUpdate,
    SpecRequest,
    ViewName,
)
from app.storage import (
    images_dir,
    relative_url,
    repository,
    sanitise_filename,
    validate_project_id,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["projects"])

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@router.post("/spec", response_model=DesignSpec)
def interpret_prompt(payload: SpecRequest) -> DesignSpec:
    """Interpret a prompt without creating a project (live Studio preview)."""
    return parse_prompt(payload.prompt)


@router.post("/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate) -> Project:
    """Create a project and immediately interpret its prompt."""
    spec = parse_prompt(payload.prompt)
    project = repository.create(payload.prompt, payload.name, spec)
    repository.append_history(project.id, "created", f"Interpreted as: {spec.object}")
    return repository.get(project.id)


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_demo: bool = Query(default=True),
) -> list[ProjectSummary]:
    return repository.list(limit=limit, offset=offset, include_demo=include_demo)


@router.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: str) -> Project:
    return repository.get(validate_project_id(project_id))


@router.patch("/projects/{project_id}", response_model=Project)
def update_project(project_id: str, payload: ProjectUpdate) -> Project:
    """Rename a project or replace its (user-edited) design specification."""
    project_id = validate_project_id(project_id)
    fields: dict[str, object] = {}
    if payload.name is not None:
        fields["name"] = payload.name
    if payload.design_spec is not None:
        fields["design_spec"] = payload.design_spec
    if not fields:
        raise ValidationError("Nothing to update.")
    project = repository.update(project_id, **fields)
    if payload.design_spec is not None:
        repository.append_history(project_id, "spec_edited", "Specification edited by user")
    return project


@router.post("/projects/{project_id}/reanalyse", response_model=DesignSpec)
def reanalyse(project_id: str) -> DesignSpec:
    """Re-derive the specification from the stored prompt, discarding edits."""
    project_id = validate_project_id(project_id)
    project = repository.get(project_id)
    return analyse_prompt(project_id, project.prompt)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str) -> None:
    repository.delete(validate_project_id(project_id))


@router.get("/projects/{project_id}/blueprint")
def get_blueprint(project_id: str) -> dict:
    """Orthographic outlines and dimensions derived from the actual mesh."""
    return blueprint_data(validate_project_id(project_id))


@router.get("/projects/{project_id}/blueprint.svg", response_class=Response)
def get_blueprint_sheet(project_id: str, theme: str = Query(default="white"),
                        download: bool = Query(default=False)) -> Response:
    """The combined third-angle blueprint sheet, as vector SVG."""
    project_id = validate_project_id(project_id)
    if theme not in ("white", "blueprint"):
        raise ValidationError("Theme must be 'white' or 'blueprint'.")

    svg = blueprint_sheet(project_id, theme)
    headers = {}
    if download:
        stem = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in repository.get(project_id).name
        ).strip("-") or "blueprint"
        headers["Content-Disposition"] = f'attachment; filename="{stem}-blueprint.svg"'
    return Response(content=svg, media_type="image/svg+xml", headers=headers)


@router.delete("/projects/{project_id}/images/{view}", response_model=Project)
def delete_view(project_id: str, view: ViewName) -> Project:
    """Remove one generated view from the project."""
    project_id = validate_project_id(project_id)
    project = repository.get(project_id)
    remaining = [image for image in project.images if image.view != view]
    if len(remaining) == len(project.images):
        raise ValidationError(f"This project has no {view.value} view.")
    for image in project.images:
        if image.view == view:
            Path(image.path).unlink(missing_ok=True)
    repository.set_images(project_id, remaining)
    repository.append_history(project_id, "view_removed", view.value)
    return repository.get(project_id)


@router.post("/projects/{project_id}/images/{view}", response_model=Project)
async def upload_reference(project_id: str, view: ViewName,
                           file: UploadFile = File(...)) -> Project:  # noqa: B008
    """Attach a user-supplied reference image as one of the views."""
    settings = get_settings()
    project_id = validate_project_id(project_id)
    project = repository.get(project_id)

    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_IMAGE_TYPES and suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValidationError(
            "Only PNG, JPEG and WebP images can be uploaded.",
            hint=f"Received '{file.content_type or suffix or 'unknown'}'.",
        )

    payload = await file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise ValidationError(
            f"That image is larger than the "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB upload limit."
        )
    if not payload:
        raise ValidationError("The uploaded file is empty.")

    target_dir = images_dir(project_id, demo=project.is_demo)
    target_dir.mkdir(parents=True, exist_ok=True)
    # The stored name comes from the view enum, never from the upload.
    destination = target_dir / f"{sanitise_filename(view.value)}.png"

    try:
        import io  # noqa: PLC0415

        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        with Image.open(io.BytesIO(payload)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            converted = opened.convert("RGB")
            converted.save(destination, format="PNG")
            width, height = converted.size
    except UnidentifiedImageError as exc:
        raise ValidationError("That file is not a readable image.") from exc

    replacement = GeneratedImage(
        view=view,
        path=str(destination),
        url=relative_url(destination) or "",
        width=width,
        height=height,
        provider="upload",
        created_at=utcnow(),
        is_reference=True,
    )
    images = [image for image in project.images if image.view != view] + [replacement]
    images.sort(key=lambda image: list(ViewName).index(image.view))
    repository.set_images(project_id, images)
    repository.append_history(project_id, "reference_uploaded", view.value)
    return repository.get(project_id)
