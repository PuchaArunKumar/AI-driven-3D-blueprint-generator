"""Shared fixtures.

Every test runs against an isolated storage directory and SQLite file, so the
suite never touches the developer's real projects.  The environment is set
before :mod:`app.config` is imported anywhere.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Point the application at a throwaway storage root *before* any app import.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="blueprint-tests-"))
os.environ["STORAGE_DIR"] = str(_TEST_ROOT)
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"
os.environ["LOG_LEVEL"] = "WARNING"
# Never let a developer's real key be spent by the test suite.
os.environ["OPENAI_API_KEY"] = ""
os.environ["STABILITY_API_KEY"] = ""
os.environ["TRELLIS_ENDPOINT"] = ""

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Base, get_engine, init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_storage() -> Iterator[None]:
    settings = get_settings()
    settings.ensure_directories()
    init_db()
    yield
    Base.metadata.drop_all(get_engine())


@pytest.fixture(autouse=True)
def _clean_projects() -> Iterator[None]:
    """Start every test with an empty projects table."""
    from app.db import ProjectRow, session_scope

    with session_scope() as session:
        session.query(ProjectRow).delete()
    yield


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def storage_root() -> Path:
    return _TEST_ROOT


@pytest.fixture
def cube_mesh():
    """A small watertight box - valid geometry for mesh and export tests."""
    import trimesh

    return trimesh.creation.box(extents=(0.4, 0.2, 0.6))


@pytest.fixture
def cube_glb(tmp_path: Path, cube_mesh) -> Path:
    path = tmp_path / "cube.glb"
    cube_mesh.export(path)
    return path


@pytest.fixture
def silhouette_images(tmp_path: Path) -> list:
    """Six synthetic views of a dark box on a white background.

    Real enough to exercise segmentation, normalisation and carving without
    invoking a diffusion model.
    """
    from PIL import Image

    from app.providers.base import ImageResult
    from app.schemas import ViewName

    directory = tmp_path / "views"
    directory.mkdir(exist_ok=True)
    results = []
    # Half-extent of the object in each view, per (columns, rows).
    boxes = {
        ViewName.FRONT: (0.30, 0.38),
        ViewName.REAR: (0.30, 0.38),
        ViewName.LEFT: (0.22, 0.38),
        ViewName.RIGHT: (0.22, 0.38),
        ViewName.TOP: (0.30, 0.22),
        ViewName.BOTTOM: (0.30, 0.22),
    }
    size = 256
    for view, (half_width, half_height) in boxes.items():
        canvas = np.full((size, size, 3), 250, dtype=np.uint8)
        x0 = int((0.5 - half_width) * size)
        x1 = int((0.5 + half_width) * size)
        y0 = int((0.5 - half_height) * size)
        y1 = int((0.5 + half_height) * size)
        canvas[y0:y1, x0:x1] = (40, 90, 160)
        path = directory / f"{view.value}.png"
        Image.fromarray(canvas).save(path)
        results.append(
            ImageResult(view=view, path=path, width=size, height=size, provider="test", seed=1)
        )
    return results


@pytest.fixture
def project(client: TestClient) -> dict:
    """A created project with an interpreted specification."""
    response = client.post(
        "/api/projects",
        json={"prompt": "A minimalist oak side table, 40 x 40 x 55 cm, CNC machined"},
    )
    assert response.status_code == 201, response.text
    return response.json()
