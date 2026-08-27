"""System status, hardware detection, capability reporting and downloads."""

from __future__ import annotations

import logging
import platform
import shutil
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app import __version__
from app.config import get_settings
from app.errors import NotFoundError
from app.jobs import manager
from app.providers.registry import get_registry
from app.schemas import CapabilityInfo, HardwareInfo, SystemStatus
from app.storage import ensure_within, raw_exports, repository, validate_project_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["system"])

#: MIME types for the formats we serve, so browsers handle them sensibly.
MEDIA_TYPES = {
    "glb": "model/gltf-binary",
    "gltf": "model/gltf+json",
    "obj": "text/plain",
    "ply": "application/octet-stream",
    "stl": "model/stl",
    "blend": "application/octet-stream",
    "step": "application/step",
    "stp": "application/step",
}


def detect_hardware() -> HardwareInfo:
    """Report the machine's compute resources, degrading where unavailable."""
    info = HardwareInfo(
        cpu=platform.processor() or platform.machine() or "unknown",
        cpu_cores=__import__("os").cpu_count() or 1,
        platform=f"{platform.system()} {platform.release()}",
    )

    try:
        import psutil  # noqa: PLC0415

        memory = psutil.virtual_memory()
        info.ram_total_gb = round(memory.total / 1e9, 2)
        info.ram_available_gb = round(memory.available / 1e9, 2)
    except ImportError:
        logger.debug("psutil not installed; RAM reporting unavailable")

    try:
        import torch  # noqa: PLC0415

        info.torch_version = torch.__version__
        info.cuda_version = torch.version.cuda
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            properties = torch.cuda.get_device_properties(0)
            info.gpu = properties.name
            info.vram_total_gb = round(properties.total_memory / 1e9, 2)
    except ImportError:
        logger.debug("torch not installed; GPU reporting unavailable")
    except Exception:
        logger.warning("GPU probe failed", exc_info=True)

    return info


def detect_capabilities() -> CapabilityInfo:
    """What this installation can actually produce, right now."""
    registry = get_registry()
    available, unavailable = registry.available_export_formats()

    diffusers_installed = True
    try:
        import diffusers  # noqa: F401,PLC0415
    except ImportError:
        diffusers_installed = False

    blender_availability = registry.blender.availability()
    return CapabilityInfo(
        export_formats=sorted(set(available)),
        unavailable_export_formats=unavailable,
        blender=blender_availability.available,
        blender_version=registry.blender.version() if blender_availability.available else None,
        freecad=registry.freecad.availability().available,
        diffusers_installed=diffusers_installed,
    )


@router.get("/system/status", response_model=SystemStatus)
def system_status() -> SystemStatus:
    """Everything the Settings page needs, in one call."""
    settings = get_settings()
    registry = get_registry()
    providers = registry.describe()

    has_image = any(p.available for p in providers if p.kind == "image")
    has_threed = any(p.available for p in providers if p.kind == "threed")

    return SystemStatus(
        status="ok" if (has_image and has_threed) else "degraded",
        version=__version__,
        hardware=detect_hardware(),
        capabilities=detect_capabilities(),
        providers=providers,
        settings=settings.redacted(),
        project_count=repository.count(),
        active_jobs=manager.active_count(),
    )


@router.get("/system/storage")
def storage_usage() -> dict:
    """Disk usage for the storage root, for the Settings page."""
    settings = get_settings()
    usage = shutil.disk_usage(settings.storage_dir)

    def directory_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    return {
        "storage_dir": str(settings.storage_dir),
        "disk_total_gb": round(usage.total / 1e9, 2),
        "disk_free_gb": round(usage.free / 1e9, 2),
        "images_bytes": directory_size(settings.images_dir or Path()),
        "models_bytes": directory_size(settings.models_dir or Path()),
        "exports_bytes": directory_size(settings.exports_dir or Path()),
        "demo_bytes": directory_size(settings.demo_dir or Path()),
    }


@router.get("/projects/{project_id}/download/{fmt}")
def download_export(project_id: str, fmt: str) -> FileResponse:
    """Serve a generated asset as a download.

    The path comes from the project's own export record and is re-checked
    against the storage root, so a crafted format string cannot escape it.
    """
    settings = get_settings()
    project_id = validate_project_id(project_id)
    project = repository.get(project_id)

    fmt = fmt.lower().lstrip(".")
    raw = raw_exports(project_id)
    if fmt not in raw:
        raise NotFoundError(
            f"This project has no {fmt.upper()} export yet.",
            hint="Export the format first from the download panel.",
        )

    path = ensure_within(Path(raw[fmt]), settings.storage_dir)
    if not path.exists():
        raise NotFoundError(f"The {fmt.upper()} file is missing from storage.")

    stem = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in (project.name or "model")
    ).strip("-") or "model"
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(fmt, "application/octet-stream"),
        filename=f"{stem}.{fmt}",
    )
