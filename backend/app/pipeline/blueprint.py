"""Technical line drawings projected from the generated mesh.

A blueprint is not a filled silhouette. It is a line drawing made of three
distinct edge families, and this module extracts all three from the real
geometry rather than tracing a raster mask:

**outline**  The view-dependent contour. An edge belongs to it when one of the
             two faces sharing it points towards the camera and the other away.
             This is what produces the outer profile *and* the interior
             contours - a wheel arch, a cut-out, the lip of a recess.

**inline**   Crease edges: a sharp dihedral angle between two faces that both
             face the camera. These are the panel lines and hard edges that
             make a drawing read as an object instead of a blob.

**hidden**   The same edges where the body occludes them. Drafting convention
             draws these dashed, so they are returned separately rather than
             discarded.

Visibility is resolved with a depth buffer rasterised from the mesh, so an edge
behind the body is correctly classified as hidden.

Coordinate frame matches the rest of the pipeline (glTF: +X right, +Y up,
+Z towards the viewer). Each view fixes a horizontal axis, a vertical axis and
a depth axis, with larger depth meaning nearer the camera.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: view -> (horizontal axis, vertical axis, depth axis)
#: Depth increases towards the camera, so front-facing means normal[depth] > 0.
VIEW_AXES: dict[str, tuple[int, int, int]] = {
    "front": (0, 1, 2),   # look along -Z from +Z: X right, Y up
    "side": (2, 1, 0),    # look along -X from +X: Z right, Y up
    "top": (0, 2, 1),     # look down -Y from +Y: X right, Z up
}

#: Faces above this dihedral angle count as a crease worth drawing.
DEFAULT_CREASE_DEGREES = 28.0

#: Drawing at full density is illegible and slow; decimate first.
DRAWING_FACE_BUDGET = 30_000

#: Depth-buffer resolution used for hidden-line removal.
DEPTH_RESOLUTION = 900

#: Depth tolerance, as a fraction of the model's depth range.
DEPTH_BIAS = 0.004

#: Light smoothing before crease detection, to suppress marching-cubes ripple.
DRAWING_SMOOTH_ITERATIONS = 6

#: Only dense reconstructions carry that ripple. Smoothing a clean low-poly
#: mesh just rounds its corners off and shrinks it, so leave those alone.
SMOOTHING_FACE_THRESHOLD = 5_000

#: Inline segments shorter than this fraction of the view's diagonal are noise.
MIN_SEGMENT_FRACTION = 0.012


@dataclass(slots=True)
class TechnicalView:
    """One projected orthographic view, in model units."""

    outline: list[list[list[float]]] = field(default_factory=list)
    inline: list[list[list[float]]] = field(default_factory=list)
    hidden: list[list[list[float]]] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

    def to_dict(self) -> dict:
        return {
            "outline": self.outline,
            "inline": self.inline,
            "hidden": self.hidden,
            "width": round(self.width, 6),
            "height": round(self.height, 6),
        }


def _prepare_for_drawing(mesh):
    """Decimate and smooth a copy so the crease detection is not swamped by noise.

    Marching-cubes surfaces carry a fine ripple whose dihedral angles read as
    thousands of spurious creases. A light Taubin pass removes the ripple while
    leaving genuine hard edges intact.
    """
    import trimesh  # noqa: PLC0415

    working = mesh.copy()
    if len(working.faces) > DRAWING_FACE_BUDGET:
        try:
            reduced = working.simplify_quadric_decimation(face_count=DRAWING_FACE_BUDGET)
            if reduced is not None and len(reduced.faces) > 0:
                working = reduced
        except Exception:
            logger.debug("Decimation unavailable for drawing", exc_info=True)
    if len(working.faces) >= SMOOTHING_FACE_THRESHOLD:
        try:
            trimesh.smoothing.filter_taubin(working, iterations=DRAWING_SMOOTH_ITERATIONS)
        except Exception:
            logger.debug("Smoothing unavailable for drawing", exc_info=True)
    return working


def _depth_buffer(points2d: np.ndarray, depth: np.ndarray, faces: np.ndarray,
                  resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise a nearest-surface depth buffer.

    Returns the buffer plus the offset/scale mapping model space to pixels.
    Rasterising per triangle over its own bounding box keeps this to one small
    vectorised operation per face rather than a per-pixel Python loop.
    """
    minimum = points2d.min(axis=0)
    span = np.maximum(points2d.max(axis=0) - minimum, 1e-9)
    scale = (resolution - 2) / span.max()

    pixels = (points2d - minimum) * scale + 1.0
    buffer = np.full((resolution, resolution), -np.inf, dtype=np.float32)

    triangles = pixels[faces]                      # (F, 3, 2)
    depths = depth[faces]                          # (F, 3)

    lower = np.floor(triangles.min(axis=1)).astype(np.int32)
    upper = np.ceil(triangles.max(axis=1)).astype(np.int32)
    np.clip(lower, 0, resolution - 1, out=lower)
    np.clip(upper, 0, resolution - 1, out=upper)

    ax, ay = triangles[:, 0, 0], triangles[:, 0, 1]
    bx, by = triangles[:, 1, 0], triangles[:, 1, 1]
    cx, cy = triangles[:, 2, 0], triangles[:, 2, 1]
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)

    for index in range(len(faces)):
        if abs(area[index]) < 1e-12:
            continue
        x0, y0 = lower[index]
        x1, y1 = upper[index]
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        inverse = 1.0 / area[index]
        w0 = ((bx[index] - grid_x) * (cy[index] - grid_y)
              - (cx[index] - grid_x) * (by[index] - grid_y)) * inverse
        w1 = ((cx[index] - grid_x) * (ay[index] - grid_y)
              - (ax[index] - grid_x) * (cy[index] - grid_y)) * inverse
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        face_depth = (w0 * depths[index, 0] + w1 * depths[index, 1]
                      + w2 * depths[index, 2])
        window = buffer[y0:y1 + 1, x0:x1 + 1]
        np.maximum(window, np.where(inside, face_depth, -np.inf), out=window)

    return buffer, minimum, scale


def _split_visible(segments: np.ndarray, depths: np.ndarray, buffer: np.ndarray,
                   minimum: np.ndarray, scale: float, bias: float,
                   samples: int = 12) -> tuple[list, list]:
    """Split each segment into visible and hidden runs against the buffer.

    Returns ``(visible, hidden)`` as lists of 2-point polylines.
    """
    if len(segments) == 0:
        return [], []

    resolution = buffer.shape[0]
    t = np.linspace(0.0, 1.0, samples)[None, :, None]           # (1, S, 1)
    points = segments[:, 0][:, None, :] * (1 - t) + segments[:, 1][:, None, :] * t
    sample_depth = depths[:, 0][:, None] * (1 - t[..., 0]) + depths[:, 1][:, None] * t[..., 0]

    pixels = (points - minimum) * scale + 1.0
    columns = np.clip(pixels[..., 0].astype(np.int32), 0, resolution - 1)
    rows = np.clip(pixels[..., 1].astype(np.int32), 0, resolution - 1)
    surface = buffer[rows, columns]

    visible_mask = sample_depth >= surface - bias

    visible: list = []
    hidden: list = []
    for index in range(len(segments)):
        start, end = segments[index]
        flags = visible_mask[index]
        # A segment is rarely half-hidden at this sampling density, so classify
        # by majority rather than emitting many slivers.
        if flags.mean() >= 0.5:
            visible.append([[float(start[0]), float(start[1])],
                            [float(end[0]), float(end[1])]])
        else:
            hidden.append([[float(start[0]), float(start[1])],
                           [float(end[0]), float(end[1])]])
    return visible, hidden


def _chain(segments: list) -> list:
    """Join 2-point segments into longer polylines where endpoints coincide.

    Fewer, longer paths mean far less SVG for the browser to draw.
    """
    if not segments:
        return []

    def key(point):
        return (round(point[0], 6), round(point[1], 6))

    remaining = dict(enumerate(segments))
    adjacency: dict = {}
    for index, segment in remaining.items():
        adjacency.setdefault(key(segment[0]), []).append(index)
        adjacency.setdefault(key(segment[1]), []).append(index)

    chains: list = []
    used: set = set()
    for start_index in list(remaining):
        if start_index in used:
            continue
        used.add(start_index)
        chain = list(remaining[start_index])

        # Extend from both ends until nothing connects.
        for end in (1, 0):
            while True:
                tip = key(chain[-1] if end else chain[0])
                nxt = None
                for candidate in adjacency.get(tip, ()):
                    if candidate not in used:
                        nxt = candidate
                        break
                if nxt is None:
                    break
                used.add(nxt)
                segment = remaining[nxt]
                other = segment[1] if key(segment[0]) == tip else segment[0]
                if end:
                    chain.append(other)
                else:
                    chain.insert(0, other)

        chains.append([[float(x), float(y)] for x, y in chain])
    return chains


def technical_view(mesh, view: str, crease_degrees: float = DEFAULT_CREASE_DEGREES,
                   resolution: int = DEPTH_RESOLUTION,
                   outline: list | None = None,
                   source_extents: np.ndarray | None = None) -> TechnicalView:
    """Project ``mesh`` into one orthographic technical view.

    Args:
        outline: pre-traced silhouette loops for this view. Supplied by
            :func:`technical_drawing`, which traces all three views at once.
        source_extents: extents of the *original* mesh. The drawing runs on a
            decimated and possibly smoothed copy, so the dimensions quoted on
            the sheet must come from the real geometry, not that copy.
    """
    if view not in VIEW_AXES:
        raise ValueError(f"unknown view '{view}'")

    horizontal, vertical, depth_axis = VIEW_AXES[view]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return TechnicalView()

    points2d = vertices[:, [horizontal, vertical]]
    depth = vertices[:, depth_axis]

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    front_facing = normals[:, depth_axis] > 0.0

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)

    outline = outline or []
    if len(adjacency) == 0:
        return TechnicalView(outline=outline)

    first, second = front_facing[adjacency[:, 0]], front_facing[adjacency[:, 1]]
    contour_mask = first != second
    crease_mask = (angles > np.radians(crease_degrees)) & first & second

    span = float(depth.max() - depth.min()) or 1.0
    bias = span * DEPTH_BIAS

    buffer, minimum, scale = _depth_buffer(points2d, depth, faces, resolution)

    def project(mask: np.ndarray):
        edges = adjacency_edges[mask]
        if len(edges) == 0:
            return np.zeros((0, 2, 2)), np.zeros((0, 2))
        return points2d[edges], depth[edges]

    contour_segments, contour_depth = project(contour_mask)
    crease_segments, crease_depth = project(crease_mask)

    # The contour is by definition on the visible boundary, but interior
    # contours (a far wheel arch) can still sit behind the body.
    contour_visible, contour_hidden = _split_visible(
        contour_segments, contour_depth, buffer, minimum, scale, bias
    )
    crease_visible, crease_hidden = _split_visible(
        crease_segments, crease_depth, buffer, minimum, scale, bias
    )

    extents = points2d.max(axis=0) - points2d.min(axis=0)
    if source_extents is not None:
        extents = np.array([source_extents[horizontal], source_extents[vertical]])
    diagonal = float(np.hypot(*extents)) or 1.0
    minimum_length = diagonal * MIN_SEGMENT_FRACTION

    def long_enough(paths: list) -> list:
        kept = []
        for path in paths:
            array = np.asarray(path)
            if np.linalg.norm(np.diff(array, axis=0), axis=1).sum() >= minimum_length:
                kept.append(path)
        return kept

    # The outer profile is traced from a raster silhouette rather than chained
    # from contour edges: that yields smooth closed loops instead of hundreds
    # of fragments. Interior contours and creases supply the inline detail.
    result = TechnicalView(
        outline=outline,
        inline=long_enough(_chain(contour_visible) + _chain(crease_visible)),
        hidden=long_enough(_chain(contour_hidden + crease_hidden)),
        width=float(extents[0]),
        height=float(extents[1]),
    )
    logger.info(
        "Technical view '%s': %d outline, %d inline, %d hidden paths",
        view, len(result.outline), len(result.inline), len(result.hidden),
    )
    return result


def technical_drawing(mesh, crease_degrees: float = DEFAULT_CREASE_DEGREES) -> dict:
    """Project all three principal views, ready for the blueprint sheet."""
    from app.providers.mesh.trimesh_processor import orthographic_outlines  # noqa: PLC0415

    drawing_mesh = _prepare_for_drawing(mesh)
    silhouettes = orthographic_outlines(drawing_mesh, resolution=384)
    source_extents = np.asarray(mesh.extents, dtype=np.float64)
    return {
        view: technical_view(
            drawing_mesh, view, crease_degrees,
            outline=silhouettes.get(view, []), source_extents=source_extents,
        ).to_dict()
        for view in VIEW_AXES
    }
