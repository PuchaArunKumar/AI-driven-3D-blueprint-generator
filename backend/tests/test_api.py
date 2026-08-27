"""HTTP API: validation, project lifecycle, jobs, exports and error handling."""

from __future__ import annotations

import io
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.providers.registry import get_registry


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    """Poll a job until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestHealthAndSystem:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_system_status_shape(self, client: TestClient) -> None:
        payload = client.get("/api/system/status").json()
        assert payload["status"] in ("ok", "degraded")
        assert payload["hardware"]["cpu_cores"] >= 1
        assert "glb" in payload["capabilities"]["export_formats"]
        assert payload["providers"]

    def test_secrets_are_never_returned(self, client: TestClient) -> None:
        """The settings blob must expose only booleans for credentials."""
        settings = client.get("/api/system/status").json()["settings"]
        assert settings["openai_api_key_set"] in (True, False)
        assert not any("key" in key and key.endswith("key") for key in settings)
        assert "openai_api_key" not in settings
        assert "stability_api_key" not in settings

    def test_storage_usage(self, client: TestClient) -> None:
        payload = client.get("/api/system/storage").json()
        assert payload["disk_total_gb"] > 0
        assert "storage_dir" in payload


class TestPromptEndpoint:
    def test_interprets_a_prompt(self, client: TestClient) -> None:
        response = client.post("/api/spec", json={"prompt": "A walnut stool 40cm tall"})
        assert response.status_code == 200
        spec = response.json()
        assert spec["materials"] == ["walnut"]
        assert spec["dimensions"]["height_mm"] == 400.0

    @pytest.mark.parametrize("prompt", ["", "  ", "a"])
    def test_rejects_empty_or_short_prompts(self, client: TestClient, prompt: str) -> None:
        response = client.post("/api/spec", json={"prompt": prompt})
        assert response.status_code == 422
        assert "message" in response.json()

    def test_rejects_missing_field(self, client: TestClient) -> None:
        assert client.post("/api/spec", json={}).status_code == 422

    def test_rejects_unknown_field(self, client: TestClient) -> None:
        response = client.post("/api/spec", json={"prompt": "A stool", "evil": 1})
        assert response.status_code == 422

    def test_rejects_oversized_prompt(self, client: TestClient) -> None:
        response = client.post("/api/spec", json={"prompt": "x" * 5000})
        assert response.status_code == 422


class TestProjectLifecycle:
    def test_create_returns_201_with_a_spec(self, client: TestClient) -> None:
        response = client.post("/api/projects", json={"prompt": "A minimalist oak stool"})
        assert response.status_code == 201
        project = response.json()
        assert project["design_spec"]["object"]
        assert project["status"] == "specified"
        assert project["history"]

    def test_create_rejects_a_blank_prompt(self, client: TestClient) -> None:
        assert client.post("/api/projects", json={"prompt": "   "}).status_code == 422

    def test_get_project(self, client: TestClient, project: dict) -> None:
        fetched = client.get(f"/api/projects/{project['id']}").json()
        assert fetched["id"] == project["id"]

    def test_get_unknown_project_is_404(self, client: TestClient) -> None:
        response = client.get(f"/api/projects/{uuid.uuid4()}")
        assert response.status_code == 404
        assert "message" in response.json()

    def test_malformed_id_is_rejected(self, client: TestClient) -> None:
        assert client.get("/api/projects/not-a-uuid").status_code == 422

    def test_path_traversal_id_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/projects/..%2F..%2Fetc%2Fpasswd")
        assert response.status_code in (404, 422)

    def test_list_projects(self, client: TestClient, project: dict) -> None:
        listed = client.get("/api/projects").json()
        assert any(item["id"] == project["id"] for item in listed)

    def test_rename(self, client: TestClient, project: dict) -> None:
        response = client.patch(f"/api/projects/{project['id']}", json={"name": "Side table"})
        assert response.status_code == 200
        assert response.json()["name"] == "Side table"

    def test_patch_with_no_fields_is_rejected(self, client: TestClient, project: dict) -> None:
        assert client.patch(f"/api/projects/{project['id']}", json={}).status_code == 422

    def test_edit_specification(self, client: TestClient, project: dict) -> None:
        spec = project["design_spec"]
        spec["color"] = "walnut brown"
        spec["materials"] = ["walnut", "brass"]
        response = client.patch(
            f"/api/projects/{project['id']}", json={"design_spec": spec}
        )
        assert response.status_code == 200
        assert response.json()["design_spec"]["color"] == "walnut brown"

    def test_reanalyse_discards_edits(self, client: TestClient, project: dict) -> None:
        spec = project["design_spec"]
        spec["object"] = "something else entirely"
        client.patch(f"/api/projects/{project['id']}", json={"design_spec": spec})
        reanalysed = client.post(f"/api/projects/{project['id']}/reanalyse").json()
        assert reanalysed["object"] != "something else entirely"

    def test_delete(self, client: TestClient, project: dict) -> None:
        assert client.delete(f"/api/projects/{project['id']}").status_code == 204
        assert client.get(f"/api/projects/{project['id']}").status_code == 404


class TestReferenceUpload:
    def _png(self, size: int = 64) -> bytes:
        import numpy as np
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8)).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_upload_attaches_a_reference_view(self, client: TestClient, project: dict) -> None:
        response = client.post(
            f"/api/projects/{project['id']}/images/front",
            files={"file": ("ref.png", self._png(), "image/png")},
        )
        assert response.status_code == 200
        images = response.json()["images"]
        assert len(images) == 1
        assert images[0]["view"] == "front"
        assert images[0]["is_reference"] is True

    def test_upload_rejects_a_non_image(self, client: TestClient, project: dict) -> None:
        response = client.post(
            f"/api/projects/{project['id']}/images/front",
            files={"file": ("payload.exe", b"MZ\x90\x00malicious", "application/x-msdownload")},
        )
        assert response.status_code == 422

    def test_upload_rejects_an_empty_file(self, client: TestClient, project: dict) -> None:
        response = client.post(
            f"/api/projects/{project['id']}/images/front",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 422

    def test_upload_rejects_an_invalid_view(self, client: TestClient, project: dict) -> None:
        response = client.post(
            f"/api/projects/{project['id']}/images/sideways",
            files={"file": ("ref.png", self._png(), "image/png")},
        )
        assert response.status_code == 422

    def test_delete_a_view(self, client: TestClient, project: dict) -> None:
        client.post(
            f"/api/projects/{project['id']}/images/front",
            files={"file": ("ref.png", self._png(), "image/png")},
        )
        response = client.delete(f"/api/projects/{project['id']}/images/front")
        assert response.status_code == 200
        assert response.json()["images"] == []

    def test_deleting_an_absent_view_is_rejected(self, client: TestClient, project: dict) -> None:
        assert client.delete(f"/api/projects/{project['id']}/images/top").status_code == 422


class TestJobs:
    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/jobs/{uuid.uuid4()}").status_code == 404

    def test_3d_without_images_fails_with_guidance(
        self, client: TestClient, project: dict
    ) -> None:
        """The job must fail cleanly, not crash, when its input is missing."""
        started = client.post("/api/generate/3d", json={"project_id": project["id"]})
        assert started.status_code == 202
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "failed"
        assert "view" in (job["error"] or "").lower()

    def test_mesh_processing_without_a_model_fails(
        self, client: TestClient, project: dict
    ) -> None:
        started = client.post("/api/process/mesh", json={"project_id": project["id"]})
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "failed"

    def test_export_without_a_model_fails(self, client: TestClient, project: dict) -> None:
        started = client.post(f"/api/export/{project['id']}", json={"formats": ["glb"]})
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "failed"

    def test_job_carries_its_stage_list(self, client: TestClient, project: dict) -> None:
        started = client.post("/api/generate/3d", json={"project_id": project["id"]}).json()
        assert [stage["key"] for stage in started["stages"]] == ["reconstruction"]

    def test_jobs_for_project(self, client: TestClient, project: dict) -> None:
        client.post("/api/generate/3d", json={"project_id": project["id"]})
        jobs = client.get(f"/api/projects/{project['id']}/jobs").json()
        assert len(jobs) >= 1

    def test_generate_for_unknown_project_is_404(self, client: TestClient) -> None:
        response = client.post("/api/generate/3d", json={"project_id": str(uuid.uuid4())})
        assert response.status_code == 404

    def test_requesting_no_views_is_rejected(self, client: TestClient, project: dict) -> None:
        response = client.post(
            "/api/generate/images", json={"project_id": project["id"], "views": []}
        )
        assert response.status_code == 422

    def test_invalid_resolution_is_rejected(self, client: TestClient, project: dict) -> None:
        response = client.post(
            "/api/generate/3d", json={"project_id": project["id"], "resolution": 4096}
        )
        assert response.status_code == 422


class TestMissingProviderHandling:
    def test_missing_api_key_yields_a_clear_message(
        self, client: TestClient, project: dict
    ) -> None:
        """With no OPENAI_API_KEY the job fails with guidance, not a traceback."""
        started = client.post(
            "/api/generate/images",
            json={"project_id": project["id"], "views": ["front"], "provider": "openai"},
        )
        assert started.status_code == 202
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "failed"
        assert "key" in (job["error"] or "").lower()
        assert "Traceback" not in (job["error"] or "")

    def test_unknown_provider_is_rejected(self, client: TestClient, project: dict) -> None:
        started = client.post(
            "/api/generate/images",
            json={"project_id": project["id"], "views": ["front"], "provider": "nonesuch"},
        )
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "failed"


class TestExportPipeline:
    """End-to-end export using an uploaded reference instead of a diffusion model."""

    @pytest.fixture
    def project_with_model(self, client: TestClient, project: dict, silhouette_images) -> dict:
        for image in silhouette_images:
            response = client.post(
                f"/api/projects/{project['id']}/images/{image.view.value}",
                files={"file": (f"{image.view.value}.png", image.path.read_bytes(), "image/png")},
            )
            assert response.status_code == 200

        started = client.post(
            "/api/generate/3d", json={"project_id": project["id"], "resolution": 48}
        )
        job = _wait_for_job(client, started.json()["id"], timeout=180)
        assert job["status"] == "completed", job.get("error")
        return client.get(f"/api/projects/{project['id']}").json()

    def test_reconstruction_produces_a_model(self, project_with_model: dict) -> None:
        assert project_with_model["model_url"]
        assert project_with_model["stats"]["triangles"] > 0
        assert project_with_model["status"] == "model_ready"

    def test_blueprint_is_derived_from_the_mesh(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        payload = client.get(f"/api/projects/{project_with_model['id']}/blueprint").json()
        assert set(payload["views"]) == {"front", "side", "top"}
        front = payload["views"]["front"]
        assert {"outline", "inline", "hidden", "width", "height"} <= set(front)
        assert front["outline"], "the front view must have a silhouette"
        assert front["width"] > 0 and front["height"] > 0
        assert payload["stats"]["triangles"] > 0

    def test_blueprint_sheet_is_valid_svg(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        import xml.etree.ElementTree as ElementTree

        response = client.get(f"/api/projects/{project_with_model['id']}/blueprint.svg")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")

        root = ElementTree.fromstring(response.text)
        assert root.tag.endswith("svg")
        # A real A3 sheet, so the scale in the title block means something.
        assert root.get("width") == "420.0mm"
        assert root.get("height") == "297.0mm"
        assert "<path" in response.text

    def test_blueprint_sheet_themes(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        base = f"/api/projects/{project_with_model['id']}/blueprint.svg"
        white = client.get(f"{base}?theme=white")
        blueprint = client.get(f"{base}?theme=blueprint")
        assert white.status_code == 200 and blueprint.status_code == 200
        assert white.text != blueprint.text

    def test_blueprint_sheet_rejects_an_unknown_theme(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        response = client.get(
            f"/api/projects/{project_with_model['id']}/blueprint.svg?theme=neon"
        )
        assert response.status_code == 422

    def test_blueprint_sheet_download_sets_a_filename(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        response = client.get(
            f"/api/projects/{project_with_model['id']}/blueprint.svg?download=true"
        )
        assert "attachment" in response.headers.get("content-disposition", "")

    def test_mesh_processing_succeeds(self, client: TestClient, project_with_model: dict) -> None:
        started = client.post(
            "/api/process/mesh", json={"project_id": project_with_model["id"]}
        )
        job = _wait_for_job(client, started.json()["id"], timeout=180)
        assert job["status"] == "completed", job.get("error")

    @pytest.mark.parametrize("fmt", ["glb", "stl", "obj", "ply"])
    def test_exports_each_format_and_serves_the_download(
        self, client: TestClient, project_with_model: dict, fmt: str
    ) -> None:
        started = client.post(
            f"/api/export/{project_with_model['id']}", json={"formats": [fmt]}
        )
        job = _wait_for_job(client, started.json()["id"], timeout=180)
        assert job["status"] == "completed", job.get("error")

        download = client.get(f"/api/projects/{project_with_model['id']}/download/{fmt}")
        assert download.status_code == 200
        assert len(download.content) > 0

    def test_downloading_an_unexported_format_is_404(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        response = client.get(f"/api/projects/{project_with_model['id']}/download/gltf")
        assert response.status_code == 404

    @pytest.mark.skipif(
        get_registry().freecad.availability().available,
        reason="FreeCAD is installed, so STEP is genuinely available",
    )
    def test_step_export_fails_cleanly_without_freecad(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        started = client.post(
            f"/api/export/{project_with_model['id']}", json={"formats": ["step"]}
        )
        job = _wait_for_job(client, started.json()["id"], timeout=120)
        assert job["status"] == "failed"
        # The reason for each rejected format is carried in the job's hint.
        assert "step" in job["hint"].lower()
        assert "freecad" in job["hint"].lower()

    @pytest.mark.skipif(
        not get_registry().blender.availability().available,
        reason="Blender is not installed on this machine",
    )
    def test_blend_export_via_blender(
        self, client: TestClient, project_with_model: dict
    ) -> None:
        started = client.post(
            f"/api/export/{project_with_model['id']}", json={"formats": ["blend"]}
        )
        job = _wait_for_job(client, started.json()["id"], timeout=300)
        assert job["status"] == "completed", job.get("error")

        download = client.get(f"/api/projects/{project_with_model['id']}/download/blend")
        assert download.status_code == 200
        # Blender 4.x+ Zstd-compresses .blend by default; older versions do not.
        assert download.content.startswith(b"BLENDER") or download.content.startswith(
            bytes([0x28, 0xB5, 0x2F, 0xFD])
        )
