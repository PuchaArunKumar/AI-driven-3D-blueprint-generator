"""FreeCAD CAD provider.

Converts a triangle mesh into a tessellated B-rep and exports STEP/IGES/BREP
by driving ``FreeCADCmd`` headlessly.

The distinction this provider is careful about:

    AI-generated mesh -> mesh cleanup -> CAD-compatible geometry

is *not* the same as a parametric CAD model.  The STEP file it writes is a
faceted solid: measurable and importable, but with no feature tree, no
sketches and no editable parameters.  The UI states this next to the download.
"""

from __future__ import annotations

import json
import logging
import subprocess
from functools import lru_cache
from pathlib import Path

from app.errors import CADToolError, ExportError
from app.providers.base import Availability, CADExporter

logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "cad_scripts" / "freecad_convert.py"

FREECAD_FORMATS = ("step", "stp", "iges", "igs", "brep")

#: FreeCAD reads these mesh formats directly.
MESH_INPUT_FORMATS = (".stl", ".obj", ".ply")


@lru_cache(maxsize=8)
def freecad_available(executable: str) -> bool:
    if not executable or not Path(executable).exists():
        return False
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("FreeCAD version probe failed for %s", executable, exc_info=True)
        return False
    return "freecad" in (completed.stdout + completed.stderr).lower()


class FreeCADProvider(CADExporter):
    """Mesh -> tessellated B-rep -> STEP / IGES / BREP."""

    name = "freecad"
    label = "FreeCAD"
    formats = FREECAD_FORMATS

    def __init__(self, executable: str, timeout: int = 300) -> None:
        self.executable = executable
        self.timeout = timeout

    def availability(self) -> Availability:
        if not self.executable:
            return Availability.missing(
                "FreeCAD was not found on this machine, so STEP export is unavailable.",
                requires=["freecad"],
            )
        if not freecad_available(self.executable):
            return Availability.missing(
                f"The configured FreeCAD executable did not respond: {self.executable}",
                requires=["freecad"],
            )
        return Availability.ok()

    def export(self, model_path: Path, output_path: Path, fmt: str) -> Path:
        fmt = fmt.lower().lstrip(".")
        if fmt not in self.formats:
            raise ExportError(f"FreeCAD cannot export the '{fmt}' format.")
        availability = self.availability()
        if not availability.available:
            raise CADToolError(
                availability.reason,
                hint="Install FreeCAD and set FREECAD_PATH in .env to enable STEP export.",
            )

        # FreeCAD's mesh importer does not read GLB, so hand it an STL.
        source = model_path
        if model_path.suffix.lower() not in MESH_INPUT_FORMATS:
            from app.providers.mesh.trimesh_processor import convert  # noqa: PLC0415

            source = output_path.with_suffix(".freecad-input.stl")
            convert(model_path, source)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.executable, str(SCRIPT_PATH),
            "--", "--input", str(source), "--output", str(output_path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.timeout, check=False, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CADToolError(
                f"FreeCAD did not finish within {self.timeout} seconds.",
                hint="Decimate the mesh before converting - B-rep conversion scales "
                     "poorly with triangle count.",
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise CADToolError("FreeCAD could not be started.", detail=str(exc)) from exc
        finally:
            if source is not model_path and source.exists():
                source.unlink(missing_ok=True)

        for line in (completed.stdout or "").splitlines():
            if line.startswith("RESULT_JSON:"):
                payload = json.loads(line[len("RESULT_JSON:"):])
                if not payload.get("ok"):
                    raise CADToolError(
                        "FreeCAD could not convert the mesh to CAD geometry.",
                        hint="Complex or non-manifold meshes often fail B-rep conversion. "
                             "Try processing the mesh first.",
                        detail=str(payload.get("error", "unknown error")),
                    )
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise ExportError("FreeCAD reported success but wrote no file.")
                return output_path

        raise CADToolError(
            "FreeCAD ran but returned no result.",
            detail=(completed.stderr or completed.stdout or "")[-2000:],
        )
