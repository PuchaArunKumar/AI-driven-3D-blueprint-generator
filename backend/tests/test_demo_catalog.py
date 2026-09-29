"""Registering the committed demo assets as projects, and the static site export."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db import ProjectRow, session_scope
from app.demo_catalog import DEMOS, adopt_demo_projects
from app.schemas import Project, ProjectStatus, ProjectSummary
from app.storage import repository

BACKEND = Path(__file__).resolve().parents[1]
DEMO = DEMOS[2]          # the chair
OTHER = DEMOS[0]         # the car


def _write_demo(root: Path, project_id: str, *, processed: bool = True,
                views: tuple[str, ...] = ("front", "top"),
                exports: tuple[str, ...] = ("glb", "stl")) -> Path:
    """A small but real demo folder: a box mesh, PNG views and exports."""
    import trimesh
    from PIL import Image

    directory = root / project_id
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "exports").mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(0.45, 0.9, 0.45))
    mesh.export(directory / ("model_processed.glb" if processed else "model.glb"))
    for view in views:
        Image.fromarray(np.full((64, 48, 3), 200, dtype=np.uint8)).save(
            directory / "images" / f"{view}.png")
    for fmt in exports:
        mesh.export(directory / "exports" / f"{project_id}.{fmt}")
    return directory


@pytest.fixture
def demo_dir():
    """The configured storage/demo, cleaned of anything a test put there."""
    root = get_settings().demo_dir
    root.mkdir(parents=True, exist_ok=True)
    created = []
    yield root, created
    for directory in created:
        shutil.rmtree(directory, ignore_errors=True)


class TestAdoption:
    def test_registers_a_demo_from_its_files(self, demo_dir) -> None:
        root, created = demo_dir
        created.append(_write_demo(root, DEMO.uuid))

        assert adopt_demo_projects() == [DEMO.uuid]
        project = repository.get(DEMO.uuid)
        assert project.is_demo and project.name == DEMO.name and project.prompt == DEMO.prompt
        assert project.status is ProjectStatus.MESH_PROCESSED
        assert project.design_spec is not None
        assert project.design_spec.dimensions.height_mm == 900.0
        assert project.stats is not None and project.stats.triangles == 12
        assert project.stats.height == pytest.approx(0.9)
        assert [image.view.value for image in project.images] == ["front", "top"]
        assert project.images[0].width == 48 and project.images[0].height == 64
        assert project.images[0].url.startswith("/files/demo/")
        assert set(project.exports) == {"glb", "stl"}
        assert project.model_url.endswith("/model_processed.glb")
        assert project.history[-1]["event"] == "demo_registered"

    def test_is_idempotent(self, demo_dir) -> None:
        root, created = demo_dir
        created.append(_write_demo(root, DEMO.uuid))
        assert adopt_demo_projects() == [DEMO.uuid]
        before = repository.count()
        assert adopt_demo_projects() == []
        assert repository.count() == before

    def test_never_touches_an_existing_row(self, demo_dir) -> None:
        root, created = demo_dir
        created.append(_write_demo(root, DEMO.uuid))
        adopt_demo_projects()
        repository.update(DEMO.uuid, name="Renamed by the user")
        assert adopt_demo_projects() == []
        assert repository.get(DEMO.uuid).name == "Renamed by the user"

    def test_raw_model_only_is_model_ready(self, demo_dir) -> None:
        root, created = demo_dir
        created.append(_write_demo(root, DEMO.uuid, processed=False, views=(), exports=()))
        adopt_demo_projects()
        project = repository.get(DEMO.uuid)
        assert project.status is ProjectStatus.MODEL_READY
        assert project.images == [] and project.exports == {}

    def test_skips_folders_without_a_model_and_unreadable_models(self, demo_dir) -> None:
        root, created = demo_dir
        empty = root / DEMO.uuid
        (empty / "images").mkdir(parents=True, exist_ok=True)
        created.append(empty)
        broken = root / OTHER.uuid
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "model_processed.glb").write_bytes(b"not a glb")
        created.append(broken)

        assert adopt_demo_projects() == []
        with session_scope() as session:
            assert session.get(ProjectRow, DEMO.uuid) is None
            assert session.get(ProjectRow, OTHER.uuid) is None
        assert (broken / "model_processed.glb").exists(), "files are never deleted"

    def test_no_demo_folder_is_a_no_op(self, tmp_path: Path) -> None:
        assert adopt_demo_projects(tmp_path / "does-not-exist") == []

    def test_runs_on_startup(self, demo_dir) -> None:
        root, created = demo_dir
        created.append(_write_demo(root, DEMO.uuid))
        from app.main import app

        with TestClient(app) as client:
            listed = client.get("/api/projects").json()
        assert [item["id"] for item in listed if item["is_demo"]] == [DEMO.uuid]
        assert listed[0]["thumbnail_url"].endswith("/images/front.png")

    def test_seed_script_uses_the_catalogue(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "seed_demo_under_test", BACKEND / "scripts" / "seed_demo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.DEMOS is DEMOS
        assert [demo.slug for demo in module.DEMOS] == ["car", "wheelchair", "chair", "wearable"]


# --------------------------------------------------------------------------
# Static site export
# --------------------------------------------------------------------------

#: Packages the static export must work without (the Pages build installs
#: only the minimal list in the script's docstring).
_BLOCKED = ("torch", "diffusers", "transformers", "rembg", "onnxruntime", "tokenizers",
            "fastapi", "starlette", "sqlalchemy", "uvicorn", "httpx", "huggingface_hub")

_RUNNER = """
import importlib.abc, runpy, sys
blocked = set(sys.argv[1].split(","))
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked:
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Blocker())
script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
"""


def _export(demo_root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, ",".join(_BLOCKED),
         str(BACKEND / "scripts" / "export_static_site.py"),
         "--out", str(out), "--demo-dir", str(demo_root), *extra],
        capture_output=True, text=True, timeout=300, check=False,
    )


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("demo-assets")
    _write_demo(root, DEMO.uuid, views=("front", "left", "top"))
    _write_demo(root, OTHER.uuid, processed=False, views=(), exports=("glb",))
    out = tmp_path_factory.mktemp("site") / "static-api"
    completed = _export(root, out)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "2 demo project(s)" in completed.stdout
    return out


class TestStaticExport:
    def test_layout(self, snapshot: Path) -> None:
        for project_id in (DEMO.uuid, OTHER.uuid):
            assert (snapshot / "projects" / f"{project_id}.json").is_file()
            for name in ("blueprint.json", "blueprint-white.svg", "blueprint-blueprint.svg"):
                assert (snapshot / "projects" / project_id / name).is_file(), name
        assert (snapshot / "files" / "demo" / DEMO.uuid / "model_processed.glb").is_file()
        assert (snapshot / "files" / "demo" / OTHER.uuid / "model.glb").is_file()

    def test_summaries_match_the_api_schema(self, snapshot: Path) -> None:
        summaries = json.loads((snapshot / "projects.json").read_text(encoding="utf-8"))
        assert [item["id"] for item in summaries] == [OTHER.uuid, DEMO.uuid]  # catalogue order
        for item in summaries:
            ProjectSummary.model_validate(item)
            assert set(item) == set(ProjectSummary.model_fields)
            assert item["is_demo"] is True
        assert summaries[1]["thumbnail_url"] == \
            f"static-api/files/demo/{DEMO.uuid}/images/front.png"
        assert summaries[0]["thumbnail_url"] is None

    def test_project_urls_are_relative_and_resolve(self, snapshot: Path) -> None:
        data = json.loads((snapshot / "projects" / f"{DEMO.uuid}.json").read_text(encoding="utf-8"))
        Project.model_validate(data)
        assert "metadata" in data and "project_metadata" not in data
        urls = [data["thumbnail_url"], data["model_url"],
                *(image["url"] for image in data["images"]), *data["exports"].values()]
        for url in urls:
            assert url.startswith("static-api/"), url
            assert (snapshot / url[len("static-api/"):]).is_file(), url
        assert [image["view"] for image in data["images"]] == ["front", "left", "top"]
        # The download convention api.ts relies on: exports/<id>.<fmt>.
        assert data["exports"]["stl"] == \
            f"static-api/files/demo/{DEMO.uuid}/exports/{DEMO.uuid}.stl"
        # Paths never leak the exporting machine's directory layout.
        assert not Path(data["model_path"]).is_absolute()
        assert all(not Path(image["path"]).is_absolute() for image in data["images"])

    def test_blueprint_is_the_real_drawing(self, snapshot: Path) -> None:
        data = json.loads(
            (snapshot / "projects" / DEMO.uuid / "blueprint.json").read_text(encoding="utf-8"))
        assert set(data) == {"views", "stats", "spec", "units"}
        assert data["units"] == "m"
        front, side, top = (data["views"][view] for view in ("front", "side", "top"))
        assert front["width"] == pytest.approx(0.45) and front["height"] == pytest.approx(0.9)
        assert side["width"] == pytest.approx(0.45) and top["height"] == pytest.approx(0.45)
        assert front["outline"], "the front view must have a silhouette"
        assert data["stats"]["triangles"] == 12

    def test_sheets_are_themed_svg(self, snapshot: Path) -> None:
        white = (snapshot / "projects" / DEMO.uuid / "blueprint-white.svg").read_text("utf-8")
        blue = (snapshot / "projects" / DEMO.uuid / "blueprint-blueprint.svg").read_text("utf-8")
        root = ElementTree.fromstring(white)
        assert root.tag.endswith("svg") and root.get("width") == "420.0mm"
        assert white != blue and DEMO.name in white

    def test_re_export_replaces_and_foreign_directories_are_refused(
        self, tmp_path: Path
    ) -> None:
        both, one = tmp_path / "both", tmp_path / "one"
        _write_demo(both, DEMO.uuid)
        _write_demo(both, OTHER.uuid)
        _write_demo(one, DEMO.uuid)
        out = tmp_path / "site"
        assert _export(both, out).returncode == 0
        assert (out / "projects" / f"{OTHER.uuid}.json").exists()
        again = _export(one, out)                # holds the marker: replaced wholesale
        assert again.returncode == 0, again.stderr
        assert not (out / "projects" / f"{OTHER.uuid}.json").exists()
        assert not (out / "files" / "demo" / OTHER.uuid).exists()
        root = one

        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("mine")
        refused = _export(root, foreign)
        assert refused.returncode != 0
        assert (foreign / "keep.txt").read_text() == "mine"
        assert not (foreign / "projects.json").exists()
