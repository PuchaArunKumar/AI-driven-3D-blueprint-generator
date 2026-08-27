"""Mesh loading, statistics, repair and the reconstruction stage."""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import GenerationError, MeshError
from app.providers.mesh.trimesh_processor import (
    TrimeshProcessor,
    load_mesh,
    mesh_statistics,
    orthographic_outlines,
)
from app.providers.threed.visual_hull import (
    VisualHullProvider,
    carve,
    extract_silhouette,
    normalise_masks,
)
from app.schemas import ViewName


class TestLoading:
    def test_loads_a_valid_glb(self, cube_glb) -> None:
        mesh = load_mesh(cube_glb)
        assert len(mesh.faces) == 12  # a box is 6 quads -> 12 triangles

    def test_missing_file_raises_a_clear_error(self, tmp_path) -> None:
        with pytest.raises(MeshError) as excinfo:
            load_mesh(tmp_path / "nope.glb")
        assert "missing" in excinfo.value.user_message.lower()

    def test_corrupt_file_raises_meshError(self, tmp_path) -> None:
        broken = tmp_path / "broken.glb"
        broken.write_bytes(b"this is definitely not a GLB")
        with pytest.raises(MeshError):
            load_mesh(broken)

    def test_empty_file_raises_meshError(self, tmp_path) -> None:
        empty = tmp_path / "empty.stl"
        empty.write_bytes(b"")
        with pytest.raises(MeshError):
            load_mesh(empty)


class TestStatistics:
    def test_statistics_match_the_geometry(self, cube_mesh, cube_glb) -> None:
        stats = mesh_statistics(cube_mesh, cube_glb)
        assert stats.vertices == len(cube_mesh.vertices)
        assert stats.triangles == len(cube_mesh.faces)
        assert stats.width == pytest.approx(0.4)
        assert stats.height == pytest.approx(0.2)
        assert stats.depth == pytest.approx(0.6)
        assert stats.is_watertight is True
        assert stats.volume == pytest.approx(0.4 * 0.2 * 0.6, rel=1e-6)
        assert stats.file_size_bytes > 0
        assert stats.file_format == "glb"

    def test_volume_is_omitted_for_open_meshes(self) -> None:
        import trimesh

        # A single triangle is not watertight, so volume is meaningless.
        open_mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
        )
        stats = mesh_statistics(open_mesh)
        assert stats.is_watertight is False
        assert stats.volume is None


class TestProcessing:
    def test_processing_produces_a_valid_glb(self, cube_glb, tmp_path) -> None:
        result = TrimeshProcessor().process(cube_glb, tmp_path / "out")
        assert result.path.exists()
        assert result.path.stat().st_size > 0
        assert result.file_format == "glb"
        assert load_mesh(result.path).faces.shape[0] > 0

    def test_metadata_reports_what_changed(self, cube_glb, tmp_path) -> None:
        result = TrimeshProcessor().process(
            cube_glb, tmp_path / "out", smooth_iterations=1, fill_holes=True
        )
        assert result.metadata["operations"]
        assert result.metadata["before"]["faces"] == 12
        assert "after" in result.metadata

    def test_duplicate_vertices_are_merged(self, tmp_path) -> None:
        import trimesh

        # Two coincident triangles: 6 vertices that should weld down to 3.
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2], [3, 4, 5]],
            process=False,
        )
        source = tmp_path / "dupes.ply"
        mesh.export(source)

        result = TrimeshProcessor().process(
            source, tmp_path / "out", smooth_iterations=0, fill_holes=False
        )
        assert result.metadata["after"]["vertices"] < 6

    def test_decimation_reduces_face_count(self, tmp_path) -> None:
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=4)  # ~5120 faces
        source = tmp_path / "sphere.glb"
        sphere.export(source)

        result = TrimeshProcessor().process(
            source, tmp_path / "out", smooth_iterations=0, target_faces=500
        )
        after = result.metadata["after"]["faces"]
        assert after <= len(sphere.faces)
        try:
            import fast_simplification  # noqa: F401
        except ImportError:
            pytest.skip("no decimation backend installed")
        # With a backend present the control must actually take effect.
        assert after <= 600, f"target_faces had no effect ({after} faces)"

    def test_processor_is_available(self) -> None:
        assert TrimeshProcessor().availability().available is True


class TestSilhouetteAndCarving:
    def test_silhouette_finds_the_object(self, silhouette_images) -> None:
        front = next(image for image in silhouette_images if image.view is ViewName.FRONT)
        mask = extract_silhouette(front.path, size=128)
        assert mask.any()
        coverage = mask.mean()
        # The synthetic box covers roughly 0.6 x 0.76 of the frame.
        assert 0.2 < coverage < 0.8

    def test_silhouette_of_a_blank_image_is_empty(self, tmp_path) -> None:
        from PIL import Image

        blank = tmp_path / "blank.png"
        Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8)).save(blank)
        assert not extract_silhouette(blank, size=128).any()

    def test_normalisation_aligns_shared_axes(self, silhouette_images) -> None:
        masks = {
            image.view: extract_silhouette(image.path, size=128)
            for image in silhouette_images
        }
        normalised = normalise_masks(masks)
        # Front and rear observe the same X and Y extents, so after alignment
        # their silhouettes must have equal vertical extent.
        def vertical_extent(mask: np.ndarray) -> int:
            rows = np.where(mask.any(axis=1))[0]
            return int(rows[-1] - rows[0]) if len(rows) else 0

        assert vertical_extent(normalised[ViewName.FRONT]) == pytest.approx(
            vertical_extent(normalised[ViewName.LEFT]), abs=2
        )

    def test_carving_produces_a_solid_volume(self, silhouette_images) -> None:
        masks = normalise_masks(
            {
                image.view: extract_silhouette(image.path, size=128)
                for image in silhouette_images
            }
        )
        volume = carve(masks, resolution=48)
        assert volume.any()
        assert 0.01 < volume.mean() < 0.9

    def test_one_bad_view_does_not_destroy_the_hull(self) -> None:
        """A strict intersection lets a single bad generation carve everything away."""
        size = 64
        good = np.zeros((size, size), dtype=bool)
        good[16:48, 16:48] = True
        # A view that saw something else entirely - a close-up, or a miss.
        bad = np.zeros((size, size), dtype=bool)
        bad[:6, :6] = True

        views = [ViewName.FRONT, ViewName.REAR, ViewName.LEFT, ViewName.RIGHT]
        masks = {view: good.copy() for view in views}
        masks[ViewName.TOP] = bad

        assert carve(masks, resolution=32).any(), "voting should tolerate one outlier"
        # With no tolerance the outlier wipes the volume out, as expected.
        assert not carve(masks, resolution=32, max_dissenting_views=0).any()

    def test_voting_is_strict_when_there_is_no_redundancy(self) -> None:
        """With fewer than four views every one of them must agree."""
        size = 64
        good = np.zeros((size, size), dtype=bool)
        good[16:48, 16:48] = True
        bad = np.zeros((size, size), dtype=bool)
        bad[:6, :6] = True
        assert not carve({ViewName.FRONT: good, ViewName.LEFT: bad}, resolution=32).any()

    def test_carving_with_disjoint_silhouettes_yields_nothing(self) -> None:
        """Non-overlapping views must carve away everything, not fake a shape."""
        # FRONT maps column -> +X and REAR maps column -> -X, so the same
        # left-hand strip in both views describes opposite ends of the object.
        size = 64
        strip = np.zeros((size, size), dtype=bool)
        strip[:, :10] = True
        volume = carve({ViewName.FRONT: strip, ViewName.REAR: strip.copy()}, resolution=32)
        assert not volume.any()


class TestVisualHullProvider:
    def test_reconstructs_a_glb_from_views(self, silhouette_images, tmp_path) -> None:
        result = VisualHullProvider().generate(
            silhouette_images, tmp_path / "model", resolution=48
        )
        assert result.path.exists()
        assert result.file_format == "glb"
        mesh = load_mesh(result.path)
        assert len(mesh.faces) > 0
        assert result.metadata["method"].startswith("multi-view silhouette")
        assert result.metadata["voxel_resolution"] == 48

    def test_reconstruction_respects_specified_dimensions(
        self, silhouette_images, tmp_path
    ) -> None:
        from app.pipeline.prompt_parser import parse_prompt

        spec = parse_prompt("A box 200 x 100 x 300 mm")
        result = VisualHullProvider().generate(
            silhouette_images, tmp_path / "model", spec=spec, resolution=48
        )
        mesh = load_mesh(result.path)
        # No specified dimension may be exceeded by the scaled model.
        assert mesh.extents[0] <= 0.200 + 1e-6
        assert mesh.extents[1] <= 0.300 + 1e-6

    def test_no_orthographic_views_is_rejected(self, tmp_path, silhouette_images) -> None:
        from app.providers.base import ImageResult

        perspective_only = [
            ImageResult(
                view=ViewName.PERSPECTIVE,
                path=silhouette_images[0].path,
                width=256, height=256, provider="test",
            )
        ]
        with pytest.raises(GenerationError) as excinfo:
            VisualHullProvider().generate(perspective_only, tmp_path / "model")
        assert "orthographic" in excinfo.value.user_message.lower()

    def test_blank_views_produce_a_helpful_error(self, tmp_path) -> None:
        from PIL import Image

        from app.providers.base import ImageResult

        blank = tmp_path / "blank.png"
        Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8)).save(blank)
        images = [
            ImageResult(view=view, path=blank, width=256, height=256, provider="test")
            for view in (ViewName.FRONT, ViewName.LEFT)
        ]
        with pytest.raises(GenerationError) as excinfo:
            VisualHullProvider().generate(images, tmp_path / "model", resolution=32)
        assert "silhouette" in excinfo.value.user_message.lower()


class TestShapEProvider:
    """The learned provider. Generation itself is too heavy for the suite, so
    these cover registration, honesty of reporting, and the failure paths."""

    def test_registered_and_marked_experimental(self) -> None:
        from app.providers.registry import get_registry

        provider = get_registry().threed_provider("shap_e")
        assert provider.name == "shap_e"
        assert provider.availability().experimental is True

    def test_reports_a_reason_when_unavailable(self) -> None:
        from app.providers.threed.shap_e import ShapEProvider

        availability = ShapEProvider().availability()
        if not availability.available:
            assert availability.reason

    def test_empty_image_list_is_rejected(self, tmp_path) -> None:
        from app.providers.threed.shap_e import ShapEProvider

        with pytest.raises(GenerationError):
            ShapEProvider().generate([], tmp_path / "model")

    def test_prefers_a_three_quarter_view(self, silhouette_images, tmp_path) -> None:
        """A perspective view carries more shape than a flat elevation."""
        from app.providers.base import ImageResult
        from app.providers.threed.shap_e import ShapEProvider

        perspective = ImageResult(
            view=ViewName.PERSPECTIVE, path=silhouette_images[0].path,
            width=256, height=256, provider="test",
        )
        chosen = ShapEProvider._pick_image([*silhouette_images, perspective])
        assert chosen.view is ViewName.PERSPECTIVE


class TestTripoSRProvider:
    """Compatibility shims and reporting. Inference is too heavy for the suite."""

    def test_registered_and_marked_experimental(self) -> None:
        from app.providers.registry import get_registry

        provider = get_registry().threed_provider("triposr")
        assert provider.name == "triposr"
        assert provider.availability().experimental is True

    def test_missing_source_is_reported_with_the_remedy(self, tmp_path) -> None:
        from app.providers.threed.triposr import TripoSRProvider

        availability = TripoSRProvider(source_path=str(tmp_path / "absent")).availability()
        assert availability.available is False
        assert "install_triposr" in availability.reason

    def test_vit_key_remap_matches_the_current_layout(self) -> None:
        """The checkpoint predates the transformers ViT rename."""
        from app.providers.threed.triposr import _remap_vit_keys

        old = {
            "image_tokenizer.model.encoder.layer.0.attention.attention.query.weight": 1,
            "image_tokenizer.model.encoder.layer.0.attention.output.dense.bias": 2,
            "image_tokenizer.model.encoder.layer.3.intermediate.dense.weight": 3,
            "image_tokenizer.model.encoder.layer.3.output.dense.weight": 4,
            "image_tokenizer.model.encoder.layer.7.layernorm_before.weight": 5,
            "decoder.layers.0.weight": 6,
        }
        new, changed = _remap_vit_keys(old)
        assert changed == 5
        assert "image_tokenizer.model.layers.0.attention.q_proj.weight" in new
        assert "image_tokenizer.model.layers.0.attention.o_proj.bias" in new
        assert "image_tokenizer.model.layers.3.mlp.fc1.weight" in new
        assert "image_tokenizer.model.layers.3.mlp.fc2.weight" in new
        assert "image_tokenizer.model.layers.7.layernorm_before.weight" in new
        # Keys outside the ViT encoder must pass through untouched.
        assert new["decoder.layers.0.weight"] == 6

    def test_marching_cubes_shim_matches_skimage(self) -> None:
        """The shim replaces a CUDA extension; it must extract a real surface."""
        import torch

        from app.providers.threed.triposr import _shim_marching_cubes

        grid = np.mgrid[:24, :24, :24].astype(np.float32)
        sphere = 8.0 - np.sqrt(((grid - 11.5) ** 2).sum(axis=0))
        vertices, faces = _shim_marching_cubes(torch.from_numpy(sphere), 0.0)
        assert len(vertices) > 100
        assert len(faces) > 100

    def test_marching_cubes_shim_handles_no_surface(self) -> None:
        import torch

        from app.providers.threed.triposr import _shim_marching_cubes

        flat = torch.full((8, 8, 8), -5.0)
        vertices, faces = _shim_marching_cubes(flat, 0.0)
        assert len(vertices) == 0 and len(faces) == 0

    def test_empty_image_list_is_rejected(self, tmp_path) -> None:
        from app.providers.threed.triposr import TripoSRProvider

        with pytest.raises(GenerationError):
            TripoSRProvider(source_path="").generate([], tmp_path / "model")


class TestBlueprintOutlines:
    def test_outlines_are_returned_for_every_plane(self, cube_mesh) -> None:
        outlines = orthographic_outlines(cube_mesh, resolution=96)
        assert set(outlines) == {"front", "side", "top"}
        for polygons in outlines.values():
            assert polygons, "each plane must yield at least one polygon"
            assert len(polygons[0]) >= 4

    def test_outline_extent_matches_the_mesh(self, cube_mesh) -> None:
        outlines = orthographic_outlines(cube_mesh, resolution=128)
        points = np.array(outlines["front"][0])
        width = points[:, 0].max() - points[:, 0].min()
        # Front plane shows X (0.4 wide); allow for rasterisation error.
        assert width == pytest.approx(0.4, abs=0.03)
