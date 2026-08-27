"""Project repository and on-disk asset layout.

All filesystem access for project artefacts goes through here, which is also
where path-traversal defence lives: every path handed back to the API is
checked to sit inside the configured storage root.
"""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.config import get_settings
from app.db import ProjectRow, session_scope, utcnow
from app.errors import NotFoundError, ValidationError
from app.schemas import (
    DesignSpec,
    GeneratedImage,
    ModelStats,
    Project,
    ProjectStatus,
    ProjectSummary,
)

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def sanitise_filename(name: str, *, default: str = "file") -> str:
    """Reduce ``name`` to a safe basename with no directory components."""
    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._-")
    cleaned = cleaned[:120]
    return cleaned or default


def validate_project_id(project_id: str) -> str:
    """Reject anything that is not a canonical UUID.

    Project ids are used to build filesystem paths, so this is the boundary
    that stops ``../`` ever reaching :func:`project_dir`.
    """
    if not isinstance(project_id, str) or not _UUID_RE.match(project_id.lower()):
        raise ValidationError("That project id is not valid.")
    return project_id.lower()


def ensure_within(path: Path, root: Path) -> Path:
    """Return ``path`` resolved, guaranteeing it lies inside ``root``."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValidationError("That path is outside the storage directory.")
    return resolved


# --------------------------------------------------------------------------
# Asset layout
# --------------------------------------------------------------------------


def project_dir(project_id: str, *, demo: bool = False) -> Path:
    """Root directory holding every artefact for one project."""
    settings = get_settings()
    validate_project_id(project_id)
    base = settings.demo_dir if demo else settings.models_dir
    assert base is not None
    return base / project_id


def images_dir(project_id: str, *, demo: bool = False) -> Path:
    return project_dir(project_id, demo=demo) / "images"


def exports_dir(project_id: str, *, demo: bool = False) -> Path:
    return project_dir(project_id, demo=demo) / "exports"


def relative_url(path: str | Path) -> str | None:
    """Map an absolute artefact path to its ``/files/...`` URL."""
    if not path:
        return None
    settings = get_settings()
    try:
        relative = Path(path).resolve().relative_to(settings.storage_dir.resolve())
    except (ValueError, OSError):
        return None
    return "/files/" + relative.as_posix()


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return utcnow()


def row_to_project(row: ProjectRow) -> Project:
    """Convert an ORM row into the API model, refreshing derived URLs."""
    images = [
        GeneratedImage(
            view=item["view"],
            path=item["path"],
            url=relative_url(item["path"]) or item.get("url", ""),
            width=item.get("width", 0),
            height=item.get("height", 0),
            provider=item.get("provider", ""),
            seed=item.get("seed"),
            created_at=_as_datetime(item.get("created_at")),
            is_reference=item.get("is_reference", False),
        )
        for item in (row.images or [])
    ]
    return Project(
        id=row.id,
        name=row.name,
        prompt=row.prompt,
        status=ProjectStatus(row.status),
        is_demo=row.is_demo,
        thumbnail_url=relative_url(row.thumbnail_path) if row.thumbnail_path else (
            images[0].url if images else None
        ),
        model_format=row.model_format,
        model_path=row.model_path,
        model_url=relative_url(row.model_path) if row.model_path else None,
        design_spec=DesignSpec.model_validate(row.design_spec) if row.design_spec else None,
        images=images,
        stats=ModelStats.model_validate(row.stats) if row.stats else None,
        exports={
            fmt: relative_url(path) or ""
            for fmt, path in (row.exports or {}).items()
            if Path(path).exists()
        },
        history=row.history or [],
        metadata=row.extra or {},
        created_at=_as_datetime(row.created_at),
        updated_at=_as_datetime(row.updated_at),
    )


def row_to_summary(row: ProjectRow) -> ProjectSummary:
    images = row.images or []
    thumbnail = row.thumbnail_path or (images[0]["path"] if images else None)
    return ProjectSummary(
        id=row.id,
        name=row.name,
        prompt=row.prompt,
        status=ProjectStatus(row.status),
        is_demo=row.is_demo,
        thumbnail_url=relative_url(thumbnail) if thumbnail else None,
        model_format=row.model_format,
        created_at=_as_datetime(row.created_at),
        updated_at=_as_datetime(row.updated_at),
    )


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------


def raw_exports(project_id: str) -> dict[str, str]:
    """Export paths on disk, as opposed to the ``/files`` URLs the API returns."""
    validate_project_id(project_id)
    with session_scope() as session:
        row = session.get(ProjectRow, project_id)
        return dict(row.exports or {}) if row else {}


class ProjectRepository:
    """CRUD for projects. Every method opens its own short transaction."""

    def create(self, prompt: str, name: str | None = None,
               design_spec: DesignSpec | None = None, *, is_demo: bool = False,
               project_id: str | None = None) -> Project:
        identifier = project_id or str(uuid.uuid4())
        validate_project_id(identifier)
        with session_scope() as session:
            row = ProjectRow(
                id=identifier,
                name=(name or self._derive_name(prompt, design_spec))[:200],
                prompt=prompt,
                status=ProjectStatus.SPECIFIED.value if design_spec
                else ProjectStatus.DRAFT.value,
                is_demo=is_demo,
                design_spec=design_spec.model_dump(mode="json") if design_spec else None,
                images=[],
                exports={},
                history=[],
                extra={},
            )
            session.add(row)
            session.flush()
            project = row_to_project(row)
        logger.info("Created project %s (%s)", identifier, project.name)
        return project

    @staticmethod
    def _derive_name(prompt: str, design_spec: DesignSpec | None) -> str:
        """Name the project after the parsed object, falling back to the prompt.

        The object phrase reads far better than the first seven words of the
        prompt, which usually truncates mid-dimension.
        """
        if design_spec and design_spec.object:
            phrase = design_spec.object.strip()
        else:
            # Cut at the first clause break, then cap the length.
            phrase = " ".join(prompt.strip().split(",")[0].split()[:8])
        return (phrase[:1].upper() + phrase[1:]) if phrase else "Untitled design"

    def get(self, project_id: str) -> Project:
        validate_project_id(project_id)
        with session_scope() as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                raise NotFoundError("That project no longer exists.")
            return row_to_project(row)

    def list(self, *, limit: int = 100, offset: int = 0,
             include_demo: bool = True) -> list[ProjectSummary]:
        with session_scope() as session:
            statement = select(ProjectRow).order_by(ProjectRow.updated_at.desc())
            if not include_demo:
                statement = statement.where(ProjectRow.is_demo.is_(False))
            rows = session.scalars(statement.limit(limit).offset(offset)).all()
            return [row_to_summary(row) for row in rows]

    def count(self) -> int:
        with session_scope() as session:
            return int(session.scalar(select(func.count()).select_from(ProjectRow)) or 0)

    def update(self, project_id: str, **fields: Any) -> Project:
        """Patch arbitrary columns; unknown keys are ignored."""
        validate_project_id(project_id)
        with session_scope() as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                raise NotFoundError("That project no longer exists.")
            for key, value in fields.items():
                if isinstance(value, (DesignSpec, ModelStats)):
                    value = value.model_dump(mode="json")
                elif key == "status" and isinstance(value, ProjectStatus):
                    value = value.value
                if hasattr(row, key):
                    setattr(row, key, value)
            row.updated_at = utcnow()
            session.flush()
            return row_to_project(row)

    def append_history(self, project_id: str, event: str, detail: str = "",
                       payload: dict[str, Any] | None = None) -> None:
        """Record a processing step. Used by the Project Details timeline."""
        validate_project_id(project_id)
        with session_scope() as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                return
            entry = {
                "event": event,
                "detail": detail,
                "at": utcnow().isoformat(),
            }
            if payload:
                entry["payload"] = payload
            # JSON columns need a new list object to be marked dirty.
            row.history = [*(row.history or []), entry]
            row.updated_at = utcnow()

    def set_images(self, project_id: str, images: list[GeneratedImage]) -> Project:
        payload = []
        for image in images:
            item = image.model_dump(mode="json")
            item["path"] = str(image.path)
            payload.append(item)
        return self.update(project_id, images=payload)

    def delete(self, project_id: str) -> None:
        """Remove the project row and every artefact it owns."""
        validate_project_id(project_id)
        settings = get_settings()
        with session_scope() as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                raise NotFoundError("That project no longer exists.")
            is_demo = row.is_demo
            session.delete(row)

        directory = project_dir(project_id, demo=is_demo)
        try:
            ensure_within(directory, settings.storage_dir)
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
        except ValidationError:
            logger.error("Refused to delete out-of-root directory for %s", project_id)
        logger.info("Deleted project %s", project_id)


repository = ProjectRepository()
