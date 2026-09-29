"""Mesh cleanup and repair built on :mod:`trimesh`.

Every operation here is a real geometric change to the mesh.  The processor
reports exactly what it did so the UI can show the before/after difference
rather than a generic "processed" badge.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from app.errors import MeshError
from app.providers.base import (
    Availability,
    MeshProcessor,
    ModelResult,
    ProgressCallback,
    _noop_progress,
)

logger = logging.getLogger(__name__)


def load_mesh(path: Path):
    """Load ``path`` as a single concatenated :class:`trimesh.Trimesh`.

    Raises:
        MeshError: when the file cannot be read or holds no triangles.
    """
    import trimesh  # noqa: PLC0415

    if not path.exists():
        raise MeshError(f"The model file is missing: {path.name}")
    try:
        loaded = trimesh.load(path, force="mesh")
    except Exception as exc:  # trimesh raises a wide range of loader errors
        raise MeshError(
            "The 3D file could not be read.",
            hint="It may be corrupt or in an unsupported format.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise MeshError("The 3D file contains no geometry.")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise MeshError("The 3D file contains no triangles.")
    return loaded


def mesh_statistics(mesh, path: Path | None = None, *, generation_time_s: float | None = None):
    """Compute the geometry statistics the viewer displays."""
    from app.schemas import ModelStats  # noqa: PLC0415 - avoids a circular import

    bounds = mesh.bounds
    extents = mesh.extents

    volume: float | None = None
    surface_area: float | None = None
    try:
        surface_area = float(mesh.area)
        if mesh.is_watertight:
            volume = abs(float(mesh.volume))
    except Exception:  # degenerate geometry - statistics stay unavailable
        logger.debug("Could not compute volume/area", exc_info=True)

    return ModelStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        triangles=int(len(mesh.faces)),
        bbox_min=[round(float(value), 6) for value in bounds[0]],
        bbox_max=[round(float(value), 6) for value in bounds[1]],
        width=round(float(extents[0]), 6),
        height=round(float(extents[1]), 6),
        depth=round(float(extents[2]), 6),
        volume=round(volume, 9) if volume is not None else None,
        surface_area=round(surface_area, 9) if surface_area is not None else None,
        is_watertight=bool(mesh.is_watertight),
        file_size_bytes=path.stat().st_size if path and path.exists() else 0,
        file_format=path.suffix.lstrip(".").lower() if path else "",
        generation_time_s=generation_time_s,
    )


def align_to_principal_axes(mesh) -> dict:
    """Rotate the mesh so its own axes line up with the world axes.

    A learned reconstruction places the object in whatever pose the source
    image implied, so a car can sit diagonally inside its axis-aligned bounding
    box. That makes the reported width/height/depth meaningless and turns the
    "front" orthographic view into a three-quarter view.

    The oriented bounding box gives the object's natural axes. The one closest
    to world up stays up (the generator already established which way is up);
    of the remaining two the longer becomes depth/length (+Z) and the shorter
    becomes width (+X).

    Returns a description of what was done, for the processing history.
    """
    import trimesh  # noqa: PLC0415

    try:
        obb = mesh.bounding_box_oriented
        transform = np.asarray(obb.primitive.transform, dtype=np.float64)
        extents = np.asarray(obb.primitive.extents, dtype=np.float64)
    except Exception:
        logger.debug("Oriented bounding box unavailable; leaving orientation as-is",
                     exc_info=True)
        return {"aligned": False}

    axes = transform[:3, :3]                       # columns are the box axes
    world_up = np.array([0.0, 1.0, 0.0])

    up_index = int(np.argmax(np.abs(axes.T @ world_up)))
    remaining = [i for i in (0, 1, 2) if i != up_index]
    length_index, width_index = sorted(remaining, key=lambda i: -extents[i])

    up = axes[:, up_index]
    if up @ world_up < 0:
        up = -up
    depth = axes[:, length_index]
    right = np.cross(up, depth)
    if np.linalg.norm(right) < 1e-9:
        return {"aligned": False}
    right /= np.linalg.norm(right)
    depth = np.cross(right, up)                    # re-orthogonalise

    # Rows map world axes onto the object's axes, so this rotates object -> world.
    rotation = np.eye(4)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = depth
    rotation[:3, :3] = rotation[:3, :3].T
    if np.linalg.det(rotation[:3, :3]) < 0:
        rotation[:3, 0] *= -1                      # keep it right-handed

    before = mesh.extents.copy()
    mesh.apply_transform(rotation)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    del trimesh

    return {
        "aligned": True,
        "extents_before": [round(float(v), 6) for v in before],
        "extents_after": [round(float(v), 6) for v in mesh.extents],
    }


def _laplacian_smooth(mesh, iterations: int) -> None:
    """Apply Taubin smoothing, which reduces noise without shrinking volume."""
    import trimesh  # noqa: PLC0415

    if iterations <= 0:
        return
    try:
        trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
    except Exception:
        logger.debug("Taubin smoothing unavailable, falling back to Humphrey", exc_info=True)
        try:
            trimesh.smoothing.filter_humphrey(mesh, iterations=iterations)
        except Exception:
            logger.warning("Smoothing failed; leaving mesh unsmoothed", exc_info=True)


def has_vertex_colours(mesh) -> bool:
    """Whether the mesh carries real per-vertex colour.

    trimesh hands back a default grey when nothing is set, so the colour array
    alone cannot answer this - ``visual.kind`` can.
    """
    return getattr(mesh.visual, "kind", None) == "vertex"


def transfer_vertex_colours(source, target) -> bool:
    """Copy colour from ``source`` onto ``target`` by nearest vertex.

    Quadric decimation discards colour attributes, so a decimated mesh exports
    with no COLOR_0 and opens grey in Blender even though the reconstruction
    was coloured. The vertices move only slightly, so nearest-neighbour lookup
    restores the appearance faithfully.
    """
    if not has_vertex_colours(source) or len(target.vertices) == 0:
        return False
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        colours = np.asarray(source.visual.vertex_colors)
        tree = cKDTree(np.asarray(source.vertices))
        _, nearest = tree.query(np.asarray(target.vertices), k=1)
        target.visual.vertex_colors = colours[nearest]
        return True
    except Exception:
        logger.warning("Could not transfer vertex colours", exc_info=True)
        return False


def _simplify(mesh, target_faces: int):
    """Reduce the triangle count, preferring trimesh's quadric decimation."""
    if target_faces >= len(mesh.faces):
        return mesh
    for method, kwargs in (
        ("simplify_quadric_decimation", {"face_count": target_faces}),
        ("simplify_quadric_decimation", {"count": target_faces}),
    ):
        function = getattr(mesh, method, None)
        if function is None:
            break
        try:
            return function(**kwargs)
        except TypeError:
            continue
        except Exception:
            logger.warning("Decimation failed; keeping the full-resolution mesh",
                           exc_info=True)
            return mesh
    logger.warning(
        "No decimation backend available - install fast-simplification to enable "
        "the target-faces control"
    )
    return mesh


class TrimeshProcessor(MeshProcessor):
    """Cleans generated meshes: welding, hole filling, smoothing, decimation."""

    name = "trimesh"
    label = "Trimesh mesh processor"

    def availability(self) -> Availability:
        try:
            import trimesh  # noqa: F401,PLC0415
        except ImportError:
            return Availability.missing("The 'trimesh' package is not installed.",
                                        requires=["trimesh"])
        return Availability.ok()

    def process(
        self,
        model_path: Path,
        output_dir: Path,
        *,
        smooth_iterations: int = 2,
        target_faces: int | None = None,
        fill_holes: bool = True,
        remove_duplicates: bool = True,
        recompute_normals: bool = True,
        align_axes: bool = True,
        progress: ProgressCallback = _noop_progress,
    ) -> ModelResult:
        started = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        progress(0.05, "Loading mesh")
        mesh = load_mesh(model_path)
        before = {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        operations: list[str] = []

        if align_axes:
            progress(0.12, "Aligning to principal axes")
            alignment = align_to_principal_axes(mesh)
            if alignment.get("aligned"):
                operations.append("aligned to principal axes")
        else:
            # The generator's own frame is authoritative (front faces +Z).
            alignment = {"aligned": False, "reason": "kept the generator's canonical frame"}

        if remove_duplicates:
            progress(0.2, "Merging duplicate vertices and removing degenerate faces")
            mesh.merge_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            operations.append("merged duplicate vertices, removed degenerate faces")

        if fill_holes:
            progress(0.4, "Filling holes")
            was_watertight = mesh.is_watertight
            try:
                mesh.fill_holes()
                if mesh.is_watertight and not was_watertight:
                    operations.append("filled holes (mesh is now watertight)")
                elif not was_watertight:
                    operations.append("attempted hole filling (mesh is still open)")
            except Exception:
                logger.warning("Hole filling failed", exc_info=True)

        if smooth_iterations:
            progress(0.55, f"Smoothing ({smooth_iterations} iterations)")
            _laplacian_smooth(mesh, smooth_iterations)
            operations.append(f"Taubin smoothing x{smooth_iterations}")

        if target_faces:
            progress(0.7, f"Decimating to ~{target_faces:,} faces")
            simplified = _simplify(mesh, target_faces)
            if simplified is not None and len(simplified.faces):
                if len(simplified.faces) < len(mesh.faces):
                    operations.append(
                        f"decimated {len(mesh.faces):,} -> {len(simplified.faces):,} faces"
                    )
                # Decimation drops colour; put it back before the mesh is lost.
                if (
                    has_vertex_colours(mesh)
                    and not has_vertex_colours(simplified)
                    and transfer_vertex_colours(mesh, simplified)
                ):
                    operations.append("restored vertex colours after decimation")
                mesh = simplified

        if recompute_normals:
            progress(0.82, "Recomputing normals")
            mesh.fix_normals()
            operations.append("recomputed vertex and face normals")

        if len(mesh.faces) == 0:
            raise MeshError(
                "Mesh processing removed every triangle.",
                hint="Try again with hole filling and smoothing disabled.",
            )

        progress(0.92, "Writing processed GLB")
        output_path = output_dir / "model_processed.glb"
        try:
            mesh.export(output_path)
        except Exception as exc:
            raise MeshError(
                "The processed mesh could not be written.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        elapsed = time.time() - started
        progress(1.0, f"Processed to {len(mesh.faces):,} triangles")
        return ModelResult(
            path=output_path,
            file_format="glb",
            provider=self.name,
            generation_time_s=elapsed,
            metadata={
                "operations": operations,
                "alignment": alignment,
                "before": before,
                "after": {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))},
                "is_watertight": bool(mesh.is_watertight),
                "has_vertex_colours": has_vertex_colours(mesh),
                "euler_number": int(mesh.euler_number),
            },
        )


def convert(model_path: Path, output_path: Path) -> Path:
    """Convert a mesh between any two formats trimesh can read and write."""
    mesh = load_mesh(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    # STL carries no colour, and ASCII PLY keeps vertex colours that binary
    # exporters sometimes drop; everything else uses trimesh's default.
    try:
        if suffix == ".stl":
            # STL is a pure triangle-soup format: strip anything non-geometric.
            if hasattr(mesh.visual, "to_color"):
                mesh.visual = mesh.visual.to_color()
            output_path.write_bytes(mesh.export(file_type="stl"))
        else:
            mesh.export(output_path)
    except Exception as exc:
        raise MeshError(
            f"Conversion to {suffix.lstrip('.').upper()} failed.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise MeshError(f"Conversion to {suffix.lstrip('.').upper()} produced an empty file.")
    return output_path


def orthographic_outlines(mesh, resolution: int = 256,
                          simplify_tolerance: float = 0.75) -> dict[str, list[list[list[float]]]]:
    """Trace the true silhouette of the mesh on each principal plane.

    Points are sampled densely over the real surface, rasterised into a mask
    and contour-traced, so concave features and interior holes survive - a
    convex hull would erase both. Returns, per plane, a list of closed
    polygons in model units:

        {"front": [[[x, y], ...], ...], "side": ..., "top": ...}

    ``front`` looks down -Z (X right, Y up), ``side`` looks down -X
    (Z right, Y up) and ``top`` looks down -Y (X right, Z up).
    """
    from scipy import ndimage  # noqa: PLC0415
    from skimage import measure  # noqa: PLC0415

    bounds = mesh.bounds
    extents = np.maximum(mesh.extents, 1e-9)

    # Dense surface sampling is vectorised, so this stays fast even for
    # meshes with hundreds of thousands of triangles.
    sample_count = int(np.clip(len(mesh.faces) * 12, 60_000, 400_000))
    try:
        points, _ = trimesh_sample(mesh, sample_count)
    except Exception:
        logger.debug("Surface sampling failed; falling back to vertices", exc_info=True)
        points = np.asarray(mesh.vertices, dtype=np.float64)

    planes = {"front": (0, 1), "side": (2, 1), "top": (0, 2)}
    outlines: dict[str, list[list[list[float]]]] = {}

    for name, (horizontal, vertical) in planes.items():
        mask = np.zeros((resolution, resolution), dtype=bool)
        # Normalise into [0, 1] on each axis, then to grid indices.
        u = (points[:, horizontal] - bounds[0][horizontal]) / extents[horizontal]
        v = (points[:, vertical] - bounds[0][vertical]) / extents[vertical]
        columns = np.clip((u * (resolution - 3)).astype(np.int32) + 1, 0, resolution - 1)
        rows = np.clip((v * (resolution - 3)).astype(np.int32) + 1, 0, resolution - 1)
        mask[rows, columns] = True

        # Close the sampling gaps between adjacent surface points, then fill
        # the interior so the outline is a solid silhouette.
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
        mask = ndimage.binary_fill_holes(mask)

        polygons: list[list[list[float]]] = []
        for contour in measure.find_contours(mask.astype(np.float32), 0.5):
            if len(contour) < 8:
                continue
            simplified = measure.approximate_polygon(contour, tolerance=simplify_tolerance)
            if len(simplified) < 4:
                continue
            # Grid indices -> model units. Contours are (row, column).
            polygon = [
                [
                    round(float(bounds[0][horizontal]
                                + (column - 1) / (resolution - 3) * extents[horizontal]), 6),
                    round(float(bounds[0][vertical]
                                + (row - 1) / (resolution - 3) * extents[vertical]), 6),
                ]
                for row, column in simplified
            ]
            polygons.append(polygon)

        # Largest outline first so the UI can draw it as the main profile.
        polygons.sort(key=len, reverse=True)
        outlines[name] = polygons[:8]
    return outlines


def trimesh_sample(mesh, count: int):
    """Sample points uniformly over the mesh surface."""
    import trimesh  # noqa: PLC0415

    return trimesh.sample.sample_surface(mesh, count)
