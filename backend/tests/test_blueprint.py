"""Technical line extraction and the combined blueprint sheet."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest

from app.pipeline.blueprint import (
    VIEW_AXES,
    TechnicalView,
    technical_drawing,
    technical_view,
)
from app.pipeline.blueprint_sheet import (
    MM_PER_UNIT,
    PAPER_WIDTH_MM,
    SHEET_WIDTH,
    render_blueprint_sheet,
)


@pytest.fixture
def box():
    """A 0.4 x 0.2 x 0.6 box - every edge is a 90 degree crease."""
    import trimesh

    return trimesh.creation.box(extents=(0.4, 0.2, 0.6))


@pytest.fixture
def sphere():
    """Smooth geometry: a silhouette but essentially no creases."""
    import trimesh

    return trimesh.creation.icosphere(subdivisions=3, radius=0.3)


class TestTechnicalView:
    def test_every_principal_view_is_produced(self, box) -> None:
        drawing = technical_drawing(box)
        assert set(drawing) == {"front", "side", "top"}

    def test_view_extents_match_the_projected_axes(self, box) -> None:
        """front shows X by Y, side shows Z by Y, top shows X by Z."""
        drawing = technical_drawing(box)
        assert drawing["front"]["width"] == pytest.approx(0.4, abs=0.02)
        assert drawing["front"]["height"] == pytest.approx(0.2, abs=0.02)
        assert drawing["side"]["width"] == pytest.approx(0.6, abs=0.02)
        assert drawing["side"]["height"] == pytest.approx(0.2, abs=0.02)
        assert drawing["top"]["width"] == pytest.approx(0.4, abs=0.02)
        assert drawing["top"]["height"] == pytest.approx(0.6, abs=0.02)

    def test_outline_is_a_closed_loop(self, box) -> None:
        front = technical_drawing(box)["front"]
        assert front["outline"], "a box must have a silhouette"
        loop = front["outline"][0]
        assert len(loop) >= 4

    def test_a_sharp_edge_between_two_visible_faces_becomes_inline(self) -> None:
        """Inline detail is a crease where both faces still face the camera.

        A box seen head-on has none - its four visible edges *are* the
        silhouette. Rotating it brings two faces into view at 90 degrees, and
        their shared edge is exactly the kind of line a drawing needs.
        """
        import trimesh

        rotated = trimesh.creation.box(extents=(0.4, 0.2, 0.6))
        rotated.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(35), [0, 1, 0])
        )
        assert len(technical_drawing(rotated)["front"]["inline"]) > 0

    def test_a_smooth_body_has_far_fewer_creases_than_a_faceted_one(self, sphere) -> None:
        import trimesh

        faceted = trimesh.creation.box(extents=(0.4, 0.2, 0.6))
        faceted.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(35), [0, 1, 0])
        )
        smooth_inline = sum(len(technical_drawing(sphere)[v]["inline"]) for v in VIEW_AXES)
        faceted_inline = sum(len(technical_drawing(faceted)[v]["inline"]) for v in VIEW_AXES)
        assert smooth_inline <= faceted_inline

    def test_paths_are_pairs_of_finite_coordinates(self, box) -> None:
        for view in technical_drawing(box).values():
            for group in ("outline", "inline", "hidden"):
                for path in view[group]:
                    array = np.asarray(path, dtype=float)
                    assert array.ndim == 2 and array.shape[1] == 2
                    assert np.isfinite(array).all()

    def test_unknown_view_is_rejected(self, box) -> None:
        with pytest.raises(ValueError):
            technical_view(box, "isometric")

    def test_empty_mesh_yields_an_empty_view(self) -> None:
        import trimesh

        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        result = technical_view(empty, "front")
        assert isinstance(result, TechnicalView)
        assert result.outline == [] and result.inline == []

    def test_occluded_edges_are_classified_hidden(self) -> None:
        """A torus hides its far side behind the near one."""
        import trimesh

        torus = trimesh.creation.torus(major_radius=0.3, minor_radius=0.1)
        drawing = technical_drawing(torus)
        total_hidden = sum(len(drawing[view]["hidden"]) for view in VIEW_AXES)
        assert total_hidden > 0, "the far side of a torus must be occluded"


class TestBlueprintSheet:
    @pytest.fixture
    def sheet(self, box) -> str:
        drawing = technical_drawing(box)
        stats = {"triangles": len(box.faces)}
        spec = {"materials": ["aluminium"], "tolerances_mm": 0.2}
        return render_blueprint_sheet(drawing, stats, spec, name="Test Box")

    def test_sheet_is_wellformed_svg(self, sheet: str) -> None:
        root = ElementTree.fromstring(sheet)
        assert root.tag.endswith("svg")

    def test_sheet_declares_physical_a3_size(self, sheet: str) -> None:
        """Physical units are what make the title-block scale meaningful."""
        root = ElementTree.fromstring(sheet)
        assert root.get("width") == f"{PAPER_WIDTH_MM}mm"
        assert root.get("viewBox") is not None
        assert pytest.approx(PAPER_WIDTH_MM / SHEET_WIDTH) == MM_PER_UNIT

    def test_sheet_carries_all_three_views_and_the_title_block(self, sheet: str) -> None:
        for label in ("FRONT ELEVATION", "SIDE ELEVATION", "PLAN"):
            assert label in sheet
        assert "Test Box" in sheet
        assert "third angle" in sheet
        assert "aluminium" in sheet

    def test_sheet_states_it_is_not_a_manufacturing_drawing(self, sheet: str) -> None:
        assert "not a certified manufacturing drawing" in sheet

    def test_themes_differ(self, box) -> None:
        drawing = technical_drawing(box)
        white = render_blueprint_sheet(drawing, {"triangles": 12}, None, theme_name="white")
        blue = render_blueprint_sheet(drawing, {"triangles": 12}, None, theme_name="blueprint")
        assert white != blue
        assert "#0e4b8f" in blue

    def test_unknown_theme_falls_back_rather_than_raising(self, box) -> None:
        drawing = technical_drawing(box)
        sheet = render_blueprint_sheet(drawing, {"triangles": 12}, None, theme_name="neon")
        assert ElementTree.fromstring(sheet).tag.endswith("svg")

    def test_name_is_escaped(self, box) -> None:
        """The project name reaches the SVG, so it must not be able to inject markup."""
        drawing = technical_drawing(box)
        sheet = render_blueprint_sheet(
            drawing, {"triangles": 12}, None, name='<script>alert("x")</script>'
        )
        assert "<script>" not in sheet
        assert "&lt;script&gt;" in sheet
        ElementTree.fromstring(sheet)  # still parses
