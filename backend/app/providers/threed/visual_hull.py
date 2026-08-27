"""Local image-to-3D reconstruction by multi-view silhouette carving.

This provider implements a *visual hull* (shape-from-silhouette) reconstruction:

1. Segment each generated view into a foreground silhouette.
2. Rescale the silhouettes so that the object's extent along each world axis
   agrees between the views that observe that axis.
3. Intersect the back-projected silhouette cones inside a voxel grid.
4. Extract a surface with marching cubes and colour it by projecting each
   vertex back into the view that best faces it.

It is a classical computer-vision method, it runs on CPU with no model
weights, and it genuinely consumes the generated multi-view images.  What it
is *not* is a learned image-to-3D model: a visual hull cannot recover concave
detail that no silhouette reveals.  The UI labels the output accordingly.

Coordinate frame (glTF convention): +X right, +Y up, +Z towards the front.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.errors import GenerationError
from app.providers.base import (
    Availability,
    ImageResult,
    ModelResult,
    ProgressCallback,
    ThreeDGenerator,
    _noop_progress,
)
from app.schemas import DesignSpec, ViewName

logger = logging.getLogger(__name__)

#: Which world axes a view's (column, row) image axes correspond to, and
#: whether each runs forward (+1) or reversed (-1) relative to the world axis.
#: ``axis`` values: 0 = X, 1 = Y, 2 = Z.
VIEW_AXES: dict[ViewName, tuple[tuple[int, int], tuple[int, int]]] = {
    # view:            (column axis, sign)  (row axis, sign)
    ViewName.FRONT:    ((0, +1), (1, -1)),
    ViewName.REAR:     ((0, -1), (1, -1)),
    ViewName.RIGHT:    ((2, -1), (1, -1)),
    ViewName.LEFT:     ((2, +1), (1, -1)),
    ViewName.TOP:      ((0, +1), (2, +1)),
    ViewName.BOTTOM:   ((0, +1), (2, -1)),
}

#: Outward normal of the camera for each carving view, used to choose which
#: image colours a given surface vertex.
VIEW_NORMALS: dict[ViewName, tuple[float, float, float]] = {
    ViewName.FRONT: (0.0, 0.0, 1.0),
    ViewName.REAR: (0.0, 0.0, -1.0),
    ViewName.RIGHT: (1.0, 0.0, 0.0),
    ViewName.LEFT: (-1.0, 0.0, 0.0),
    ViewName.TOP: (0.0, 1.0, 0.0),
    ViewName.BOTTOM: (0.0, -1.0, 0.0),
}

#: Fraction of the frame the normalised silhouette should span.
TARGET_FILL = 0.86

#: A carve that removed little, on a result with no dominant axis, is almost
#: always the three-quarter-views failure: every silhouette looks alike, so the
#: intersection is a rounded block.
#:
#: This is reported, not enforced. A genuinely cuboid product (a crate, a
#: enclosure) legitimately fills its bounding box, and refusing to return it
#: would be worse than flagging it.
BLOCKY_OCCUPANCY = 0.30
BLOCKY_ASPECT = 1.35


# --------------------------------------------------------------------------
# Silhouette extraction
# --------------------------------------------------------------------------


#: rembg matting models, best quality first. Whichever one is present and
#: passes the self-check is used; they differ mainly in size and download cost
#: (u2net ~176 MB, silueta ~44 MB, u2netp ~4.6 MB).
MATTING_MODELS = ("u2net", "silueta", "u2netp")

#: A product fills a meaningful part of its frame. Anything smaller is matting
#: noise or a speck of background texture, not the object.
MIN_COVERAGE = 0.02
#: A mask covering more than this failed to separate object from background.
MAX_COVERAGE = 0.85


@lru_cache(maxsize=1)
def _matting_session():
    """Load the rembg matting model once, or return None if unavailable.

    Text-to-image models rarely honour "plain white background": real output
    has vignetted walls, studio floors and panel seams that no colour model
    separates from a dark product. A matting network solves exactly this, so
    it is preferred when installed. It is optional - the classical fallback
    keeps the pipeline working without it.
    """
    try:
        from PIL import Image  # noqa: PLC0415
        from rembg import new_session, remove  # noqa: PLC0415
    except ImportError:
        logger.info("rembg is not installed; using classical silhouette extraction")
        return None

    probe = Image.new("RGB", (64, 64), (255, 255, 255))

    for model in MATTING_MODELS:
        try:
            session = new_session(model)
        except Exception:
            logger.debug("rembg model %s could not be created", model, exc_info=True)
            continue

        # Self-check. rembg builds a session even when its weights are missing
        # or truncated, and then returns arbitrary masks - which would silently
        # corrupt every reconstruction. A blank frame contains no object, so
        # anything but a near-empty result means this model is unusable.
        try:
            alpha = np.asarray(remove(probe, session=session).convert("RGBA"))[:, :, 3]
            spurious = float((alpha > 127).mean())
        except Exception:
            logger.debug("rembg model %s failed its self-check", model, exc_info=True)
            continue

        if spurious > 0.05:
            logger.info(
                "rembg model '%s' rejected: %.0f%% foreground on a blank frame "
                "(weights missing or corrupt)", model, spurious * 100,
            )
            continue

        logger.info("Using rembg matting model '%s' for silhouette extraction", model)
        return session

    logger.warning(
        "No usable rembg model (tried: %s) - falling back to classical silhouette "
        "extraction, which is noticeably weaker on busy backgrounds. Delete any "
        "partial files in ~/.u2net and let them re-download to restore it.",
        ", ".join(MATTING_MODELS),
    )
    return None


def matting_available() -> bool:
    """Whether the learned matting backend is usable right now."""
    return _matting_session() is not None


def _matting_mask(image_path: Path, size: int) -> np.ndarray | None:
    """Silhouette from the rembg matting model, or None if it is unavailable."""
    session = _matting_session()
    if session is None:
        return None

    from PIL import Image  # noqa: PLC0415
    from rembg import remove  # noqa: PLC0415

    try:
        with Image.open(image_path) as opened:
            source = opened.convert("RGB").resize((size, size), Image.LANCZOS)
        cut_out = remove(source, session=session)
    except Exception:
        logger.warning("Matting failed for %s; falling back", image_path.name, exc_info=True)
        return None

    alpha = np.asarray(cut_out.convert("RGBA"))[:, :, 3]
    return alpha > 127


def extract_silhouette(image_path: Path, size: int = 256) -> np.ndarray:
    """Segment the product from the background.

    Prefers the learned matting model and falls back to a classical
    background-subtraction + watershed method when it is not installed.

    Returns:
        Boolean array of shape ``(size, size)`` - True where the object is.
        An all-False result means segmentation failed; the caller must treat
        that as "no silhouette", never as "empty object".
    """
    from scipy import ndimage  # noqa: PLC0415

    mask = _matting_mask(image_path, size)
    if mask is not None and mask.any():
        # Matting can leave specks; keep only the object itself.
        largest = _largest_component(mask, ndimage)
        if largest is not None and _usable(largest, image_path.name):
            return largest
    return _classical_silhouette(image_path, size)


def _usable(mask: np.ndarray, name: str) -> bool:
    """Reject a mask that found nothing, or swallowed the whole frame."""
    coverage = float(mask.mean())
    if MIN_COVERAGE <= coverage <= MAX_COVERAGE:
        return True
    logger.info("Discarding silhouette for %s: coverage %.3f outside [%.3f, %.2f]",
                name, coverage, MIN_COVERAGE, MAX_COVERAGE)
    return False


def _classical_silhouette(image_path: Path, size: int = 256) -> np.ndarray:
    """Background subtraction seeded watershed - the no-model fallback."""
    from PIL import Image  # noqa: PLC0415
    from scipy import ndimage  # noqa: PLC0415

    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB").resize((size, size), Image.LANCZOS)
    array = np.asarray(rgb).astype(np.float32) / 255.0

    # Studio backdrops are rarely a flat colour - they have vertical gradients,
    # a wall/floor seam, or a vignette. A single global background colour
    # therefore flags the gradient itself as foreground. Instead estimate the
    # background twice: once per row (from the left/right edge bands) and once
    # per column (from the top/bottom bands). A pixel is foreground only if it
    # differs from *both*, which lets each estimate absorb the gradient running
    # along its own axis, and lets the floor be cancelled by the column
    # estimate while the wall is cancelled by the row estimate.
    band = max(2, size // 32)

    side_pixels = np.concatenate([array[:, :band, :], array[:, -band:, :]], axis=1)
    row_background = np.median(side_pixels, axis=1)[:, None, :]        # (size, 1, 3)
    cap_pixels = np.concatenate([array[:band, :, :], array[-band:, :, :]], axis=0)
    column_background = np.median(cap_pixels, axis=0)[None, :, :]      # (1, size, 3)

    row_distance = np.linalg.norm(array - row_background, axis=2)
    column_distance = np.linalg.norm(array - column_background, axis=2)

    def _threshold(distances: np.ndarray) -> float:
        """Adapt to how textured the backdrop already is."""
        centre = float(np.median(distances))
        spread = float(np.median(np.abs(distances - centre)))
        return max(0.13, centre + 4.0 * spread)

    row_threshold = _threshold(
        np.concatenate([row_distance[:, :band], row_distance[:, -band:]], axis=1)
    )
    column_threshold = _threshold(
        np.concatenate([column_distance[:band, :], column_distance[-band:, :]], axis=0)
    )

    # Colour distance alone still flags smooth wall gradients and panel seams,
    # so it is used only to *seed* a watershed. The watershed itself runs on
    # edge strength: a gradient has low edge strength, so the background basin
    # floods straight through it, while the object's sharp outline holds the
    # boundary. That is what separates a dark car from a dark upper wall.
    from skimage.filters import sobel  # noqa: PLC0415
    from skimage.segmentation import watershed  # noqa: PLC0415

    deviation = np.minimum(row_distance, column_distance)
    confident = deviation > max(row_threshold, column_threshold)
    if not confident.any():
        return np.zeros((size, size), dtype=bool)

    # Seed foreground only where the deviation is strong and away from the
    # frame edge; seed background on a ring around the border.
    inset = max(2, size // 12)
    central = np.zeros((size, size), dtype=bool)
    central[inset:-inset, inset:-inset] = True

    candidate = deviation[confident & central]
    markers = np.zeros((size, size), dtype=np.int32)
    markers[:band, :] = markers[-band:, :] = 1
    markers[:, :band] = markers[:, -band:] = 1
    if candidate.size:
        strong = float(np.quantile(candidate, 0.60))
        markers[confident & central & (deviation >= strong)] = 2
    if not (markers == 2).any():
        return np.zeros((size, size), dtype=bool)

    luminance = array @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mask = watershed(sobel(luminance), markers) == 2

    # Remove speckle, then close the object's interior.
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)

    mask = _largest_component(mask, ndimage)
    if mask is None:
        return np.zeros((size, size), dtype=bool)

    if not _usable(mask, image_path.name):
        return np.zeros((size, size), dtype=bool)
    return mask


def _largest_component(mask: np.ndarray, ndimage) -> np.ndarray | None:
    """Keep the biggest blob, preferring one that does not touch the border.

    A backdrop that survived thresholding reaches the frame edge; the product
    usually does not. When everything touches the border (a cropped or
    full-bleed object) the largest blob is kept regardless.
    """
    labels, count = ndimage.label(mask)
    if count == 0:
        return None

    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)

    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    interior = [
        (sizes[index - 1], index)
        for index in range(1, count + 1)
        if index not in border_labels
    ]
    if interior:
        largest_interior, chosen = max(interior)
        # A border-touching blob that dwarfs every interior one is the object
        # itself running off the frame, not the backdrop.
        largest_overall = float(sizes.max())
        if largest_interior < 0.25 * largest_overall:
            chosen = int(np.argmax(sizes)) + 1
    else:
        chosen = int(np.argmax(sizes)) + 1

    return ndimage.binary_fill_holes(labels == chosen)


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    return int(row_indices[0]), int(row_indices[-1]), int(col_indices[0]), int(col_indices[-1])


def _place(mask: np.ndarray, row_extent: float, col_extent: float) -> np.ndarray:
    """Centre ``mask`` and resize it to span the given normalised extents."""
    from PIL import Image  # noqa: PLC0415

    size = mask.shape[0]
    box = _bounding_box(mask)
    if box is None:
        return np.zeros_like(mask)
    top, bottom, left, right = box

    cropped = Image.fromarray((mask[top:bottom + 1, left:right + 1] * 255).astype(np.uint8))
    target_rows = max(1, int(round(row_extent * size)))
    target_cols = max(1, int(round(col_extent * size)))
    resized = np.asarray(cropped.resize((target_cols, target_rows), Image.NEAREST)) > 127

    placed = np.zeros_like(mask)
    row_start = (size - target_rows) // 2
    col_start = (size - target_cols) // 2
    placed[row_start:row_start + target_rows, col_start:col_start + target_cols] = resized
    return placed


def normalise_masks(masks: dict[ViewName, np.ndarray]) -> dict[ViewName, np.ndarray]:
    """Make the object's per-axis extent agree across views.

    Independently generated images place the product at arbitrary scales and
    offsets.  Intersecting those silhouettes directly erodes the hull to
    nothing.  This measures each object's extent along every world axis, takes
    the median per axis, and rescales every silhouette to that shared size.
    """
    observations: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for view, mask in masks.items():
        box = _bounding_box(mask)
        if box is None:
            continue
        top, bottom, left, right = box
        size = mask.shape[0]
        (col_axis, _), (row_axis, _) = VIEW_AXES[view]
        observations[col_axis].append((right - left + 1) / size)
        observations[row_axis].append((bottom - top + 1) / size)

    # Median extent per axis, renormalised so the largest axis fills the frame.
    extents: dict[int, float] = {}
    for axis in (0, 1, 2):
        values = observations[axis]
        extents[axis] = float(np.median(values)) if values else 0.5
    largest = max(extents.values()) or 1.0
    extents = {axis: TARGET_FILL * value / largest for axis, value in extents.items()}

    normalised: dict[ViewName, np.ndarray] = {}
    for view, mask in masks.items():
        (col_axis, _), (row_axis, _) = VIEW_AXES[view]
        normalised[view] = _place(mask, extents[row_axis], extents[col_axis])
    return normalised


# --------------------------------------------------------------------------
# Carving
# --------------------------------------------------------------------------


def carve(masks: dict[ViewName, np.ndarray], resolution: int,
          max_dissenting_views: int | None = None) -> np.ndarray:
    """Back-project the silhouettes and combine them into an occupancy volume.

    A textbook visual hull intersects every silhouette, so a single bad view -
    a generation that came back as a close-up, or a segmentation that clipped
    the object - carves the whole model away. With four or more views this
    instead keeps a voxel that is inside all but ``max_dissenting_views`` of
    them, which tolerates one outlier without meaningfully loosening the hull.
    With fewer views there is no redundancy to spend, so the intersection stays
    strict.

    Args:
        masks: silhouette per view, all the same square shape.
        resolution: edge length of the cubic voxel grid.
        max_dissenting_views: how many views may disagree. Defaults to 1 when
            at least four views are present, otherwise 0.

    Returns:
        Boolean array of shape ``(resolution, resolution, resolution)`` indexed
        as ``[x, y, z]``.
    """
    usable = {view: mask for view, mask in masks.items() if mask.any()}
    if not usable:
        return np.zeros((resolution,) * 3, dtype=bool)

    if max_dissenting_views is None:
        max_dissenting_views = 1 if len(usable) >= 4 else 0
    required = max(1, len(usable) - max_dissenting_views)

    axis_values = (np.arange(resolution) + 0.5) / resolution
    grid_x, grid_y, grid_z = np.meshgrid(
        axis_values, axis_values, axis_values, indexing="ij"
    )
    world = (grid_x, grid_y, grid_z)

    # uint8 keeps the tally small; there are never more than seven views.
    agreement = np.zeros((resolution,) * 3, dtype=np.uint8)
    for view, mask in usable.items():
        (col_axis, col_sign), (row_axis, row_sign) = VIEW_AXES[view]
        size = mask.shape[0]

        column = world[col_axis] if col_sign > 0 else 1.0 - world[col_axis]
        row = world[row_axis] if row_sign > 0 else 1.0 - world[row_axis]

        col_index = np.clip((column * size).astype(np.int32), 0, size - 1)
        row_index = np.clip((row * size).astype(np.int32), 0, size - 1)
        agreement += mask[row_index, col_index]

    logger.info("Carving with %d view(s), requiring agreement from %d", len(usable), required)
    return agreement >= required


def _blocky_warning(volume: np.ndarray, occupancy: float) -> str:
    """Flag a carve that looks like a block rather than an object.

    Returns an empty string when the result looks plausible. The test is
    deliberately conservative: it needs *both* a carve that removed little and
    a result with no dominant axis, because a legitimately cuboid product
    trips either one on its own.
    """
    occupied = np.argwhere(volume)
    if len(occupied) == 0:
        return ""
    extents = occupied.max(axis=0) - occupied.min(axis=0) + 1
    aspect = float(extents.max() / max(extents.min(), 1))
    if occupancy > BLOCKY_OCCUPANCY and aspect < BLOCKY_ASPECT:
        return (
            "The carved hull is a near-cubic block filling "
            f"{occupancy:.0%} of the volume, which usually means the views are "
            "three-quarter product shots rather than the flat head-on elevations "
            "carving needs. Switch the 3D backend to 'auto' or 'triposr', or "
            "upload true front/side/top references."
        )
    return ""


def _sample_colour(
    points: np.ndarray,
    normals: np.ndarray,
    images: dict[ViewName, np.ndarray],
) -> np.ndarray:
    """Project each vertex into the best-facing view and read its colour."""
    colours = np.full((len(points), 3), 200, dtype=np.uint8)
    if not images:
        return colours

    view_list = list(images)
    normal_matrix = np.array([VIEW_NORMALS[view] for view in view_list], dtype=np.float32)
    # For every vertex pick the view whose camera most directly faces it.
    affinity = normals @ normal_matrix.T
    best = np.argmax(affinity, axis=1)

    for index, view in enumerate(view_list):
        selected = best == index
        if not selected.any():
            continue
        image = images[view]
        size = image.shape[0]
        (col_axis, col_sign), (row_axis, row_sign) = VIEW_AXES[view]

        column = points[selected, col_axis]
        row = points[selected, row_axis]
        if col_sign < 0:
            column = 1.0 - column
        if row_sign < 0:
            row = 1.0 - row
        col_index = np.clip((column * size).astype(np.int32), 0, size - 1)
        row_index = np.clip((row * size).astype(np.int32), 0, size - 1)
        colours[selected] = image[row_index, col_index]
    return colours


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class VisualHullProvider(ThreeDGenerator):
    """Multi-view silhouette carving - the default local 3D reconstruction."""

    name = "visual_hull"
    label = "Multi-view silhouette carving (local)"

    def availability(self) -> Availability:
        missing: list[str] = []
        for module in ("numpy", "scipy", "skimage", "trimesh", "PIL"):
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        if missing:
            return Availability.missing(
                "Missing packages: " + ", ".join(missing),
                requires=missing,
            )
        return Availability.ok()

    def generate(
        self,
        images: list[ImageResult],
        output_dir: Path,
        *,
        spec: DesignSpec | None = None,
        resolution: int = 128,
        progress: ProgressCallback = _noop_progress,
    ) -> ModelResult:
        import trimesh  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from skimage import measure  # noqa: PLC0415

        started = time.time()
        carving_views = [image for image in images if image.view in VIEW_AXES]
        if not carving_views:
            raise GenerationError(
                "Silhouette carving needs at least one orthographic view "
                "(front, rear, left, right, top or bottom).",
                hint="Generate a front and a side view, then run 3D generation again.",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_size = min(320, max(128, resolution * 2))

        progress(0.05, f"Segmenting {len(carving_views)} views")
        masks: dict[ViewName, np.ndarray] = {}
        colour_images: dict[ViewName, np.ndarray] = {}
        for index, image in enumerate(carving_views):
            mask = extract_silhouette(image.path, size=mask_size)
            if mask.any():
                masks[image.view] = mask
            with Image.open(image.path) as opened:
                colour_images[image.view] = np.asarray(
                    opened.convert("RGB").resize((mask_size, mask_size), Image.LANCZOS)
                )
            progress(0.05 + 0.25 * (index + 1) / len(carving_views),
                     f"Segmented {image.view.value} view")

        if not masks:
            raise GenerationError(
                "No product silhouette could be separated from the background in any view.",
                hint="Regenerate the views - the images need a clear object on a plain "
                     "background. Uploading a reference image with a clean background "
                     "also works.",
            )

        progress(0.35, "Aligning silhouettes across views")
        masks = normalise_masks(masks)

        progress(0.45, f"Carving {resolution}^3 voxel volume")
        volume = carve(masks, resolution)
        occupancy = float(volume.mean())
        logger.info("Visual hull occupancy: %.4f from %d views", occupancy, len(masks))

        if not volume.any():
            raise GenerationError(
                "The silhouettes did not overlap, so no solid volume remained.",
                hint="This usually means the views show the object at very different "
                     "scales. Try regenerating the views, or generate fewer views "
                     "(front and one side are enough).",
            )

        warning = _blocky_warning(volume, occupancy)
        if warning:
            logger.warning("%s", warning)

        progress(0.6, "Extracting surface (marching cubes)")
        # Pad so the surface closes at the volume boundary.
        padded = np.pad(volume.astype(np.float32), 1, mode="constant", constant_values=0.0)
        try:
            vertices, faces, normals, _ = measure.marching_cubes(padded, level=0.5)
        except (ValueError, RuntimeError) as exc:
            raise GenerationError(
                "Surface extraction failed on the carved volume.",
                detail=str(exc),
            ) from exc

        # Marching cubes works in padded voxel index space; map back to [0, 1].
        vertices = (vertices - 1.0) / float(resolution)

        progress(0.75, "Projecting colours from source views")
        colours = _sample_colour(vertices, normals, colour_images)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colours,
            process=True,
        )
        if mesh.is_empty or len(mesh.faces) == 0:
            raise GenerationError("Surface extraction produced an empty mesh.")

        progress(0.85, "Scaling to specification")
        _apply_scale(mesh, spec)

        progress(0.92, "Writing GLB")
        model_path = output_dir / "model.glb"
        mesh.export(model_path)

        elapsed = time.time() - started
        progress(1.0, f"Reconstructed {len(mesh.faces):,} triangles in {elapsed:.1f}s")
        return ModelResult(
            path=model_path,
            file_format="glb",
            provider=self.name,
            generation_time_s=elapsed,
            metadata={
                "method": "multi-view silhouette carving (visual hull)",
                "segmentation": "matting (rembg u2net)" if matting_available()
                else "classical background subtraction + watershed",
                "views_used": sorted(view.value for view in masks),
                "voxel_resolution": resolution,
                "voxel_occupancy": round(occupancy, 5),
                "warning": warning,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            },
        )


def _apply_scale(mesh, spec: DesignSpec | None) -> None:
    """Centre the mesh and scale it to the specified physical size.

    Units are metres, matching the glTF convention.  When the specification
    gives no dimensions the model is normalised to a 1 m bounding box so the
    viewer always frames it sensibly.
    """
    mesh.apply_translation(-mesh.bounding_box.centroid)
    extents = mesh.extents.copy()
    extents[extents <= 0] = 1e-6

    target = None
    if spec is not None:
        dimensions = spec.dimensions
        # length -> Z (depth), width -> X, height -> Y
        requested = (dimensions.width_mm, dimensions.height_mm, dimensions.length_mm)
        if any(value for value in requested):
            known = [
                (value / 1000.0) / extents[axis]
                for axis, value in enumerate(requested)
                if value
            ]
            # Uniform scale keeps the reconstructed proportions intact; use the
            # most conservative factor so no specified dimension is exceeded.
            target = min(known)

    if target is None:
        target = 1.0 / float(extents.max())
    mesh.apply_scale(target)
