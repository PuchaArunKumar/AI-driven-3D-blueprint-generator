"""Blender CAD provider.

Drives a local Blender installation in ``--background`` mode to import the
generated mesh, repair it, assign materials and save a native ``.blend`` file.
Blender is invoked through :mod:`subprocess` with an argument list - never a
shell string - so nothing from a user prompt can reach a command line.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from app.errors import CADToolError, ExportError
from app.providers.base import Availability, CADExporter

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "cad_scripts" / "process_model.py"

#: Formats Blender exports for us, beyond what trimesh already handles.
BLENDER_FORMATS = ("blend", "glb", "gltf", "obj", "stl", "ply", "fbx")


@lru_cache(maxsize=8)
def blender_version(executable: str) -> str | None:
    """Return the version string of the Blender at ``executable``, or None."""
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Blender version probe failed for %s", executable, exc_info=True)
        return None
    match = re.search(r"Blender\s+(\S+)", completed.stdout or "")
    return match.group(1) if match else None


class BlenderProvider(CADExporter):
    """Mesh -> Blender-native ``.blend`` (and Blender-side exports)."""

    name = "blender"
    label = "Blender"
    formats = BLENDER_FORMATS

    def __init__(self, executable: str, timeout: int = 300) -> None:
        self.executable = executable
        self.timeout = timeout

    def availability(self) -> Availability:
        if not self.executable:
            return Availability.missing(
                "Blender was not found on this machine.",
                requires=["blender"],
            )
        if not Path(self.executable).exists():
            return Availability.missing(
                f"The configured Blender path does not exist: {self.executable}",
                requires=["blender"],
            )
        if blender_version(self.executable) is None:
            return Availability.missing(
                "The configured Blender executable did not respond to --version.",
                requires=["blender"],
            )
        return Availability.ok()

    def version(self) -> str | None:
        return blender_version(self.executable)

    def _run(self, arguments: list[str]) -> dict[str, object]:
        """Execute the background script and parse its RESULT_JSON line."""
        command = [
            self.executable,
            "--background",
            "--factory-startup",
            "--python", str(SCRIPT_PATH),
            "--",
            *arguments,
        ]
        logger.info("Running Blender: %s", " ".join(command[:6]))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CADToolError(
                f"Blender did not finish within {self.timeout} seconds.",
                hint="Try processing a decimated mesh, or raise BLENDER_TIMEOUT.",
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise CADToolError(
                "Blender could not be started.",
                hint="Check the Blender path in Settings.",
                detail=str(exc),
            ) from exc

        output = completed.stdout or ""
        for line in output.splitlines():
            if line.startswith("RESULT_JSON:"):
                payload = json.loads(line[len("RESULT_JSON:"):])
                if not payload.get("ok"):
                    raise CADToolError(
                        "Blender could not process the model.",
                        detail=str(payload.get("error", "unknown error")),
                    )
                return payload

        logger.error("Blender produced no result line. stderr=%s", (completed.stderr or "")[:2000])
        raise CADToolError(
            "Blender ran but returned no result.",
            detail=(completed.stderr or output)[-2000:],
        )

    def export(self, model_path: Path, output_path: Path, fmt: str) -> Path:
        fmt = fmt.lower().lstrip(".")
        if fmt not in self.formats:
            raise ExportError(f"Blender cannot export the '{fmt}' format.")
        availability = self.availability()
        if not availability.available:
            raise CADToolError(availability.reason,
                               hint="Install Blender or set BLENDER_PATH in .env.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = ["--input", str(model_path)]
        if fmt == "blend":
            arguments += ["--output", str(output_path)]
        else:
            arguments += ["--export", str(output_path)]

        payload = self._run(arguments)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ExportError(
                f"Blender reported success but wrote no {fmt.upper()} file.",
                detail=json.dumps(payload)[:1000],
            )
        logger.info("Blender wrote %s (%d bytes)", output_path.name, output_path.stat().st_size)
        return output_path
