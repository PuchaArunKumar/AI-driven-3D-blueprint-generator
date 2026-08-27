"""Format conversion, capability reporting and CAD provider behaviour."""

from __future__ import annotations

import pytest

from app.errors import CADToolError, ExportError, MeshError
from app.providers.cad.blender import BlenderProvider
from app.providers.cad.freecad import FreeCADProvider
from app.providers.mesh.trimesh_processor import convert, load_mesh
from app.providers.registry import get_registry

#: An uncompressed .blend opens with this magic.
BLEND_MAGIC = b"BLENDER"
#: Blender 4.x+ compresses .blend files with Zstandard by default.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class TestConversion:
    @pytest.mark.parametrize("suffix", [".glb", ".obj", ".stl", ".ply"])
    def test_converts_to_each_trimesh_format(self, cube_glb, tmp_path, suffix: str) -> None:
        output = tmp_path / f"model{suffix}"
        convert(cube_glb, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_converted_file_is_loadable_geometry(self, cube_glb, tmp_path) -> None:
        """An export is only real if it can be read back as the same shape."""
        output = tmp_path / "roundtrip.stl"
        convert(cube_glb, output)
        reloaded = load_mesh(output)
        assert len(reloaded.faces) == 12
        assert reloaded.extents == pytest.approx([0.4, 0.2, 0.6], abs=1e-5)

    def test_ply_roundtrip_preserves_geometry(self, cube_glb, tmp_path) -> None:
        output = tmp_path / "roundtrip.ply"
        convert(cube_glb, output)
        assert load_mesh(output).extents == pytest.approx([0.4, 0.2, 0.6], abs=1e-5)

    def test_converting_a_missing_source_fails_cleanly(self, tmp_path) -> None:
        with pytest.raises(MeshError):
            convert(tmp_path / "absent.glb", tmp_path / "out.stl")

    def test_unsupported_target_format_fails_cleanly(self, cube_glb, tmp_path) -> None:
        with pytest.raises(MeshError):
            convert(cube_glb, tmp_path / "model.xyzzy")


class TestCapabilityReporting:
    def test_trimesh_formats_are_always_available(self) -> None:
        available, _ = get_registry().available_export_formats()
        for fmt in ("glb", "gltf", "obj", "ply", "stl"):
            assert fmt in available

    def test_unavailable_formats_carry_a_reason(self) -> None:
        """A format must never be offered without the backend to produce it."""
        available, unavailable = get_registry().available_export_formats()
        assert set(available).isdisjoint(unavailable)
        for fmt, reason in unavailable.items():
            assert reason, f"{fmt} was marked unavailable with no explanation"

    def test_step_requires_freecad(self) -> None:
        registry = get_registry()
        available, unavailable = registry.available_export_formats()
        if registry.freecad.availability().available:
            assert "step" in available
        else:
            assert "step" in unavailable


class TestBlenderProvider:
    def test_rejects_a_format_it_cannot_write(self, cube_glb, tmp_path) -> None:
        provider = BlenderProvider(executable="", timeout=30)
        with pytest.raises(ExportError):
            provider.export(cube_glb, tmp_path / "out.step", "step")

    def test_missing_executable_is_reported_not_crashed(self, cube_glb, tmp_path) -> None:
        provider = BlenderProvider(executable="", timeout=30)
        availability = provider.availability()
        assert availability.available is False
        assert "blender" in availability.reason.lower()
        with pytest.raises(CADToolError):
            provider.export(cube_glb, tmp_path / "out.blend", "blend")

    def test_nonexistent_path_is_reported(self) -> None:
        provider = BlenderProvider(executable="/no/such/blender", timeout=30)
        assert provider.availability().available is False

    @pytest.mark.skipif(
        not get_registry().blender.availability().available,
        reason="Blender is not installed on this machine",
    )
    def test_exports_a_real_blend_file(self, cube_glb, tmp_path) -> None:
        output = tmp_path / "model.blend"
        get_registry().blender.export(cube_glb, output, "blend")
        assert output.exists()
        header = output.read_bytes()[:7]
        assert header.startswith(BLEND_MAGIC) or header.startswith(ZSTD_MAGIC), header

    @pytest.mark.skipif(
        not get_registry().blender.availability().available,
        reason="Blender is not installed on this machine",
    )
    def test_blender_reports_its_version(self) -> None:
        assert get_registry().blender.version()


class TestFreeCADProvider:
    def test_missing_freecad_is_reported_not_crashed(self, cube_glb, tmp_path) -> None:
        provider = FreeCADProvider(executable="", timeout=30)
        availability = provider.availability()
        assert availability.available is False
        assert "step" in availability.reason.lower() or "freecad" in availability.reason.lower()
        with pytest.raises(CADToolError):
            provider.export(cube_glb, tmp_path / "out.step", "step")

    def test_rejects_a_format_it_cannot_write(self, cube_glb, tmp_path) -> None:
        provider = FreeCADProvider(executable="", timeout=30)
        with pytest.raises(ExportError):
            provider.export(cube_glb, tmp_path / "out.glb", "glb")
