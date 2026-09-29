"""Application configuration.

All settings are read from environment variables (optionally via a ``.env``
file at the repository root).  Secrets are *never* hard-coded and never
returned by the public API - :meth:`Settings.redacted` is used for that.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

ImageProviderName = Literal["diffusers", "openai", "stability"]
ThreeDProviderName = Literal[
    "auto", "visual_hull", "triposr", "shap_e", "trellis_api", "text2voxel"
]


def _default_blender_path() -> str:
    """Best-effort discovery of a Blender executable.

    Checks ``PATH`` first, then the conventional Windows/macOS/Linux install
    locations.  Returns an empty string when Blender cannot be found, which
    the CAD provider reports as an unavailable capability rather than an error.
    """
    found = shutil.which("blender")
    if found:
        return found

    candidates: list[Path] = []
    program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    blender_root = Path(program_files) / "Blender Foundation"
    if blender_root.is_dir():
        # Prefer the highest version directory, e.g. "Blender 5.2".
        for child in sorted(blender_root.iterdir(), reverse=True):
            candidates.append(child / "blender.exe")
    candidates += [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def _default_freecad_path() -> str:
    """Best-effort discovery of a FreeCAD command-line executable."""
    for name in ("FreeCADCmd", "freecadcmd", "FreeCAD", "freecad"):
        found = shutil.which(name)
        if found:
            return found

    program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    root = Path(program_files)
    if root.is_dir():
        for child in sorted(root.glob("FreeCAD*"), reverse=True):
            candidate = child / "bin" / "FreeCADCmd.exe"
            if candidate.is_file():
                return str(candidate)
    for candidate in (
        Path("/Applications/FreeCAD.app/Contents/Resources/bin/FreeCADCmd"),
        Path("/usr/bin/FreeCADCmd"),
        Path("/usr/local/bin/FreeCADCmd"),
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


class Settings(BaseSettings):
    """Typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- server
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --------------------------------------------------------------- storage
    storage_dir: Path = REPO_ROOT / "storage"
    images_dir: Path | None = None
    models_dir: Path | None = None
    exports_dir: Path | None = None
    tmp_dir: Path | None = None
    demo_dir: Path | None = None
    database_url: str = ""

    # ------------------------------------------------------------- providers
    image_provider: ImageProviderName = "diffusers"
    threed_provider: ThreeDProviderName = "auto"

    # local diffusers
    diffusers_model: str = "stabilityai/sd-turbo"
    diffusers_device: str = "auto"  # auto | cuda | cpu
    diffusers_steps: int = 4
    diffusers_guidance: float = 0.0
    image_width: int = 512
    image_height: int = 512

    # hosted image providers
    openai_api_key: str = ""
    openai_image_model: str = "dall-e-3"
    stability_api_key: str = ""
    stability_model: str = "stable-image/generate/core"

    # prompt understanding
    semantic_prompt: bool = True

    # local learned image-to-3D (TripoSR) - installed by scripts/install_triposr.py
    triposr_path: Path = REPO_ROOT / "backend" / "vendor" / "TripoSR"
    triposr_resolution: int = 256

    # local learned image-to-3D (Shap-E)
    shap_e_steps: int = 64
    shap_e_guidance: float = 3.0

    # local text-to-3D (Text2Voxel-64) - installed by scripts/install_text2voxel.py.
    # A relative path is taken from the repository root. Steps/guidance left
    # unset use the values the model was trained with (its meta.json).
    text2voxel_path: Path = REPO_ROOT / "backend" / "models" / "text2voxel"
    text2voxel_steps: int | None = Field(default=None, ge=1, le=1000)
    text2voxel_guidance: float | None = Field(default=None, ge=0.0, le=20.0)

    # hosted image-to-3D provider (e.g. a self-hosted TRELLIS/SLAT endpoint)
    trellis_endpoint: str = ""
    trellis_api_key: str = ""

    # ------------------------------------------------------------------- cad
    blender_path: str = Field(default_factory=_default_blender_path)
    freecad_path: str = Field(default_factory=_default_freecad_path)
    blender_timeout: int = 300

    # ------------------------------------------------------------ generation
    voxel_resolution: int = 128
    max_upload_bytes: int = 25 * 1024 * 1024
    job_timeout: int = 1800

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _expand(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("text2voxel_path", mode="before")
    @classmethod
    def _repo_relative(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    @field_validator("text2voxel_steps", "text2voxel_guidance", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        # ``TEXT2VOXEL_STEPS=`` in a .env file means "use the model's default".
        return None if isinstance(value, str) and not value.strip() else value

    # ------------------------------------------------------------- derived
    def model_post_init(self, __context: object) -> None:
        self.images_dir = self.images_dir or self.storage_dir / "images"
        self.models_dir = self.models_dir or self.storage_dir / "models"
        self.exports_dir = self.exports_dir or self.storage_dir / "exports"
        self.tmp_dir = self.tmp_dir or self.storage_dir / "tmp"
        self.demo_dir = self.demo_dir or self.storage_dir / "demo"
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.storage_dir / 'projects.db').as_posix()}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_directories(self) -> None:
        """Create every storage directory the application writes to."""
        for directory in (
            self.storage_dir,
            self.images_dir,
            self.models_dir,
            self.exports_dir,
            self.tmp_dir,
            self.demo_dir,
        ):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Settings safe to expose over the API - secrets become booleans."""
        return {
            "image_provider": self.image_provider,
            "threed_provider": self.threed_provider,
            "semantic_prompt": self.semantic_prompt,
            "diffusers_model": self.diffusers_model,
            "diffusers_device": self.diffusers_device,
            "diffusers_steps": self.diffusers_steps,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "voxel_resolution": self.voxel_resolution,
            "triposr_path": str(self.triposr_path),
            "text2voxel_path": str(self.text2voxel_path),
            "text2voxel_steps": self.text2voxel_steps,
            "text2voxel_guidance": self.text2voxel_guidance,
            "blender_path": self.blender_path,
            "freecad_path": self.freecad_path,
            "storage_dir": str(self.storage_dir),
            "images_dir": str(self.images_dir),
            "models_dir": str(self.models_dir),
            "exports_dir": str(self.exports_dir),
            "openai_api_key_set": bool(self.openai_api_key),
            "stability_api_key_set": bool(self.stability_api_key),
            "trellis_endpoint_set": bool(self.trellis_endpoint),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
