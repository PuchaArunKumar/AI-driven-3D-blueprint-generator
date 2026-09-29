"""Text2Voxel-64: sampler maths, reference parity, meshing, availability and the text path.

The model-file tests run against whatever is installed in
``backend/models/text2voxel`` (the untrained dev placeholder or the trained
release - same formats) and are skipped when nothing is installed. Everything
else uses stub providers, so the orchestration is tested without a model.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.errors import GenerationError, JobCancelled, ProviderUnavailableError
from app.providers.base import (
    Availability,
    ImageResult,
    ModelResult,
    TextTo3DGenerator,
    ThreeDGenerator,
    _noop_progress,
    is_text_conditioned,
)
from app.providers.registry import get_registry
from app.providers.threed.text2voxel import (
    MODEL_FILES,
    Text2VoxelProvider,
    ddim_sample,
    ddim_timesteps,
    mean_pool,
    prepare_field,
    scale_to_spec,
    voxels_to_mesh,
)
from app.schemas import DesignSpec, Dimensions

MODEL_DIR = Path(os.environ.get(
    "TEXT2VOXEL_TEST_MODEL_DIR",
    Path(__file__).resolve().parents[1] / "models" / "text2voxel",
))
MODEL_INSTALLED = all((MODEL_DIR / name).is_file() for name in MODEL_FILES)
needs_model = pytest.mark.skipif(
    not MODEL_INSTALLED,
    reason=f"Text2Voxel model files not installed in {MODEL_DIR} "
           "(python scripts/install_text2voxel.py)",
)


def cosine_alphas_cumprod(steps: int = 1000, s: float = 0.008) -> np.ndarray:
    """The training schedule, copied from training/text2voxel/k2_train.py."""
    t = np.linspace(0, steps, steps + 1) / steps
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas = np.clip(f[1:] / f[:-1], 1e-4, 0.9999)
    return np.cumprod(alphas).astype(np.float64)


ALPHAS = cosine_alphas_cumprod()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# --------------------------------------------------------------------------
# Sampler maths
# --------------------------------------------------------------------------


class TestSampler:
    def test_timesteps_match_the_training_kernel(self) -> None:
        times = ddim_timesteps(1000, 50)
        assert times[0] == 999 and times[-1] == 0 and len(times) == 50
        np.testing.assert_array_equal(times, np.linspace(999, 0, 50).round().astype(int))

    def test_timesteps_round_half_to_even(self) -> None:
        # 999 * 1/6 = 166.5: numpy rounds to the even 166, not 167.
        assert ddim_timesteps(1000, 7)[-2] == 166

    def test_rejects_impossible_step_counts(self) -> None:
        with pytest.raises(ValueError):
            ddim_timesteps(1000, 0)
        with pytest.raises(ValueError):
            ddim_timesteps(1000, 1001)

    @staticmethod
    def _perfect_prior(target: np.ndarray):
        """An exact v-predictor for a distribution concentrated on ``target``."""

        def prior(x, t, cond):
            a = ALPHAS[int(t[0])]
            eps = (x - math.sqrt(a) * target) / math.sqrt(1 - a)
            return (math.sqrt(a) * eps - math.sqrt(1 - a) * target).astype(np.float32)

        return prior

    def test_a_perfect_prior_lands_on_its_target(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.uniform(-3, 3, 256).astype(np.float32)
        noise = rng.standard_normal(256).astype(np.float32)
        cond = rng.standard_normal(384).astype(np.float32)
        for guidance in (1.0, 3.0):
            out = ddim_sample(self._perfect_prior(target), cond, noise, ALPHAS,
                              steps=50, guidance=guidance)
            assert np.abs(out - target).max() < 1e-4

    def test_x0_is_clipped(self) -> None:
        target = np.zeros(256, dtype=np.float32)
        target[0], target[1] = 10.0, -9.0
        out = ddim_sample(self._perfect_prior(target), np.zeros(384, np.float32),
                          np.zeros(256, np.float32), ALPHAS, steps=10, guidance=3.0,
                          x0_clip=6.0)
        assert out[0] == pytest.approx(6.0, abs=1e-4)
        assert out[1] == pytest.approx(-6.0, abs=1e-4)

    def test_guidance_batches_conditional_first_then_null(self) -> None:
        calls = []

        def prior(x, t, cond):
            calls.append((x.copy(), t.copy(), cond.copy()))
            # conditional row -> 1, null row -> 0
            return np.stack([np.ones(256), np.zeros(256)]).astype(np.float32)

        cond = np.full(384, 0.05, dtype=np.float32)
        noise = np.linspace(-1, 1, 256).astype(np.float32)
        guidance = 3.0
        out = ddim_sample(prior, cond, noise, ALPHAS, steps=1, guidance=guidance)

        (x, t, conds), = calls
        assert x.shape == (2, 256) and t.shape == (2,) and conds.shape == (2, 384)
        assert t.dtype == np.float32 and t.tolist() == [999.0, 999.0]
        np.testing.assert_array_equal(conds[0], cond)
        assert not conds[1].any()
        # v = v_u + g (v_c - v_u) = g; one step ends at a_prev = 1, so x = x0.
        a = ALPHAS[999]
        expected = np.clip(math.sqrt(a) * noise - math.sqrt(1 - a) * guidance, -6, 6)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_reports_every_step_and_stops_when_the_callback_raises(self) -> None:
        seen = []
        prior_calls = []

        def prior(x, t, cond):
            prior_calls.append(int(t[0]))
            return np.zeros_like(x)

        def on_step(index, total):
            seen.append((index, total))
            if index == 3:
                raise JobCancelled()

        with pytest.raises(JobCancelled):
            ddim_sample(prior, np.zeros(384), np.zeros(256), ALPHAS, steps=20,
                        guidance=3.0, on_step=on_step)
        assert seen == [(0, 20), (1, 20), (2, 20), (3, 20)]
        assert len(prior_calls) == 4

    def test_mean_pool_ignores_masked_tokens_and_normalises(self) -> None:
        hidden = np.zeros((1, 3, 384), dtype=np.float32)
        hidden[0, 0, 0], hidden[0, 1, 0], hidden[0, 2, 1] = 1.0, 3.0, 100.0
        pooled = mean_pool(hidden, np.array([[1, 1, 0]]))
        assert pooled.shape == (1, 384)
        assert pooled[0, 0] == pytest.approx(1.0)
        assert pooled[0, 1] == 0.0
        assert np.linalg.norm(pooled[0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Surface extraction and scale
# --------------------------------------------------------------------------


class TestMeshing:
    def test_sphere_is_closed_with_the_right_volume(self) -> None:
        centres = (np.indices((64, 64, 64)).transpose(1, 2, 3, 0) + 0.5) / 64
        occupancy = (np.linalg.norm(centres - 0.5, axis=-1) < 0.3).astype(np.float32)
        mesh = voxels_to_mesh(occupancy, None, 0.5)
        assert mesh.is_watertight
        assert mesh.volume > 0, "normals must face outwards"
        assert mesh.volume == pytest.approx(4 / 3 * math.pi * 0.3**3, rel=0.05)

    def test_voxel_frame_and_indexing(self) -> None:
        """Voxel [x][y][z] spans [i, i+1)/64; a binary box's surface sits on those faces."""
        occupancy = np.zeros((64, 64, 64), dtype=np.float32)
        occupancy[8:24, 0:40, 30:34] = 1.0     # touches the y = 0 boundary
        mesh = voxels_to_mesh(occupancy, None, 0.5)
        np.testing.assert_allclose(mesh.bounds[0], np.array([8, 0, 30]) / 64, atol=1e-6)
        np.testing.assert_allclose(mesh.bounds[1], np.array([24, 40, 34]) / 64, atol=1e-6)
        assert mesh.is_watertight

    def test_colour_comes_from_the_solid_side(self) -> None:
        occupancy = np.zeros((64, 64, 64), dtype=np.float32)
        occupancy[20:44, 20:44, 20:44] = 1.0
        rgb = np.zeros((3, 64, 64, 64), dtype=np.float32)
        rgb[2] = 1.0                                  # blue everywhere...
        rgb[:, 20:44, 20:44, 20:44] = 0.0
        rgb[0, 20:44, 20:44, 20:44] = 1.0             # ...red inside the shape
        mesh = voxels_to_mesh(occupancy, rgb.reshape(1, 3, 64, 64, 64), 0.5)
        colours = np.asarray(mesh.visual.vertex_colors)
        assert (colours[:, 0] == 255).all() and (colours[:, 2] == 0).all()

    def test_an_empty_grid_is_a_clear_error(self) -> None:
        with pytest.raises(GenerationError, match="empty shape"):
            voxels_to_mesh(np.full((64, 64, 64), 0.2, dtype=np.float32), None, 0.5)
        with pytest.raises(GenerationError, match="empty shape"):
            prepare_field(np.zeros((64, 64, 64), dtype=np.float32), 0.5)

    def test_floaters_are_dropped_and_the_body_kept(self) -> None:
        """Same rule as the browser's mesher.ts: 26-connected blobs under 8 voxels go."""
        occupancy = np.zeros((64, 64, 64), dtype=np.float32)
        occupancy[10:30, 10:30, 10:30] = 0.9           # the body, 8000 voxels
        occupancy[50, 50, 50] = 0.9                    # a 1-voxel speck
        occupancy[40:42, 40:42, 40:42] = 0.9           # 8 voxels: kept
        occupancy[45, 5, 5] = occupancy[46, 6, 6] = 0.9  # corner-joined pair: one blob of 2
        shape = prepare_field(occupancy, 0.5)
        assert shape.iso == 0.5
        assert shape.occupied_voxels == 8000 + 8
        assert shape.field[50, 50, 50] == 0.0 and shape.field[45, 5, 5] == 0.0
        assert shape.field[41, 41, 41] == pytest.approx(0.9)
        assert shape.warnings == [
            "Dropped 2 floating fragment(s) (3 voxels) smaller than the minimum part size."
        ]

    def test_a_near_empty_grid_is_surfaced_lower_with_a_warning(self) -> None:
        """The untrained placeholder's output: nothing reaches 0.5, so the level drops."""
        centres = (np.indices((64, 64, 64)).transpose(1, 2, 3, 0) + 0.5) / 64
        distance = np.linalg.norm(centres - 0.5, axis=-1)
        occupancy = np.clip(0.45 - distance, 0.0, None).astype(np.float32)  # peak < 0.45
        shape = prepare_field(occupancy, 0.5)
        assert shape.iso < 0.5
        # At least FALLBACK_VOXELS; a few more when distances tie at the cut.
        assert 4096 <= shape.occupied_voxels <= 4096 + 200
        assert len(shape.warnings) == 1 and shape.warnings[0].startswith("Only 0 voxels reached")
        mesh = voxels_to_mesh(shape.field, None, float(np.nextafter(np.float32(shape.iso), 0)))
        assert mesh.is_watertight and mesh.volume > 0

    @staticmethod
    def _box():
        import trimesh

        mesh = trimesh.creation.box(extents=(2.0, 1.0, 4.0))
        mesh.apply_translation([5.0, 3.0, -2.0])
        return mesh

    @pytest.mark.parametrize(("dimensions", "rule", "extents"), [
        (Dimensions(height_mm=500), "height", (1.0, 0.5, 2.0)),
        (Dimensions(height_mm=500, length_mm=9000), "height", (1.0, 0.5, 2.0)),
        (Dimensions(length_mm=1000, width_mm=300), "footprint", (0.5, 0.25, 1.0)),
        (Dimensions(width_mm=2000), "footprint", (1.0, 0.5, 2.0)),
        (Dimensions(), "default", (0.5, 0.25, 1.0)),
    ])
    def test_uniform_scale_rules(self, dimensions, rule, extents) -> None:
        mesh = self._box()
        info = scale_to_spec(mesh, DesignSpec(object="box", dimensions=dimensions))
        assert info["rule"].startswith(rule)
        np.testing.assert_allclose(mesh.extents, extents, atol=1e-9)
        assert mesh.bounds[0][1] == pytest.approx(0.0, abs=1e-9), "base sits on y = 0"
        assert (mesh.bounds[0][0] + mesh.bounds[1][0]) == pytest.approx(0.0, abs=1e-9)
        assert (mesh.bounds[0][2] + mesh.bounds[1][2]) == pytest.approx(0.0, abs=1e-9)
        assert (info["width_m"], info["height_m"], info["depth_m"]) == pytest.approx(extents)

    def test_no_spec_means_one_metre(self) -> None:
        mesh = self._box()
        scale_to_spec(mesh, None)
        assert max(mesh.extents) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


def _fake_model_dir(root: Path, meta: dict | str | None) -> Path:
    for name in MODEL_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real model")
    if meta is not None:
        (root / "meta.json").write_text(meta if isinstance(meta, str) else json.dumps(meta))
    return root


class TestAvailability:
    def test_missing_files_are_named_with_the_install_hint(self, tmp_path: Path) -> None:
        availability = Text2VoxelProvider(tmp_path / "nothing").availability()
        assert not availability.available
        assert "decoder.onnx" in availability.reason
        assert "minilm/tokenizer.json" in availability.reason
        assert "install_text2voxel.py" in availability.reason

    def test_one_missing_file_is_enough(self, tmp_path: Path) -> None:
        root = _fake_model_dir(tmp_path, {"name": "Text2Voxel-64"})
        (root / "prior.onnx").unlink()
        availability = Text2VoxelProvider(root).availability()
        assert not availability.available
        assert "prior.onnx" in availability.reason and "decoder.onnx" not in availability.reason

    def test_placeholder_model_is_available_but_says_so(self, tmp_path: Path) -> None:
        root = _fake_model_dir(tmp_path, {"name": "Text2Voxel-64", "smoke": True})
        availability = Text2VoxelProvider(root).availability()
        assert availability.available
        assert "placeholder" in availability.reason and "--force" in availability.reason

    def test_trained_model_is_simply_available(self, tmp_path: Path) -> None:
        root = _fake_model_dir(tmp_path, {"name": "Text2Voxel-64", "smoke": False})
        availability = Text2VoxelProvider(root).availability()
        assert availability.available and availability.reason == ""

    def test_unreadable_meta(self, tmp_path: Path) -> None:
        root = _fake_model_dir(tmp_path, "{ not json")
        availability = Text2VoxelProvider(root).availability()
        assert not availability.available and "meta.json" in availability.reason

    def test_missing_package(self, tmp_path: Path, monkeypatch) -> None:
        import importlib

        real = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "tokenizers":
                raise ImportError("no tokenizers")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        root = _fake_model_dir(tmp_path, {"name": "Text2Voxel-64"})
        availability = Text2VoxelProvider(root).availability()
        assert not availability.available
        assert "tokenizers" in availability.reason and availability.requires == ["tokenizers"]

    def test_corrupt_files_fail_cleanly(self, tmp_path: Path) -> None:
        meta = {"name": "Text2Voxel-64",
                "diffusion": {"alphas_cumprod": [0.5] * 4, "train_steps": 4}}
        root = _fake_model_dir(tmp_path / "model", meta)
        provider = Text2VoxelProvider(root)
        with pytest.raises(ProviderUnavailableError) as raised:
            provider.generate_from_text("a chair", tmp_path / "out", seed=1)
        assert "could not be loaded" in raised.value.user_message
        assert "--force" in raised.value.hint

    def test_registered_and_text_conditioned(self) -> None:
        registry = get_registry()
        provider = registry.threed_provider("text2voxel")
        assert provider.name == "text2voxel"
        assert provider.label == "Text2Voxel-64 (text-to-3D, local)"
        assert is_text_conditioned(provider)
        assert not is_text_conditioned(registry.threed_provider("visual_hull"))
        names = {info.name for info in registry.describe() if info.kind == "threed"}
        assert "text2voxel" in names

    def test_images_are_not_an_input_it_accepts(self, tmp_path: Path) -> None:
        provider = Text2VoxelProvider(tmp_path)
        with pytest.raises(GenerationError, match="prompt text"):
            provider.generate([], tmp_path)

    def test_settings(self) -> None:
        from app.config import REPO_ROOT, Settings

        settings = Settings(_env_file=None, text2voxel_path="some/relative/dir",
                            text2voxel_steps="", text2voxel_guidance="")
        assert settings.text2voxel_path == (REPO_ROOT / "some" / "relative" / "dir").resolve()
        assert settings.text2voxel_steps is None and settings.text2voxel_guidance is None
        assert Settings(_env_file=None, threed_provider="text2voxel").threed_provider == \
            "text2voxel"


# --------------------------------------------------------------------------
# The real model files (placeholder or trained)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_provider():
    provider = Text2VoxelProvider(MODEL_DIR)
    yield provider
    provider.release()


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))


def _first_non_empty(provider, prompt, output_dir, spec=None, steps=None, seeds=range(1, 9)):
    """Generate with the first seed that yields a surface.

    Near-empty grids are surfaced at a lower level (``prepare_field``), so only
    an all-zero grid still fails - kept as a safety net, since the test needs
    one real mesh whatever model is installed.
    """
    last = None
    for seed in seeds:
        try:
            return provider.generate_from_text(prompt, output_dir, spec=spec, seed=seed,
                                               steps=steps)
        except GenerationError as exc:
            if "empty" not in exc.user_message:
                raise
            last = exc
    raise AssertionError(f"no seed in {list(seeds)} produced a surface: {last}")


@needs_model
class TestRealModel:
    def test_reference_parity(self, real_provider, meta) -> None:
        """The sampler reproduces the training kernel's DDIM, step for step.

        ``meta.reference`` was recorded on the training machine, and the int8
        MiniLM gives slightly different embeddings on different CPUs
        (measured: 8e-3 per component between Kaggle and a Windows laptop,
        which 50 guided steps amplify to ~8e-2 in the latent). So the sampler
        is checked tightly against an independent re-implementation fed the
        SAME condition, and the drift from the training machine is held only
        to a loose bound that still catches a wrong encoder or schedule.
        """
        import onnxruntime as ort

        reference = meta["reference"]
        diffusion = meta["diffusion"]
        noise = np.asarray(reference["noise"], dtype=np.float32)
        cond, latent = real_provider.sample_latent(
            reference["prompt"], noise,
            steps=diffusion["sample_steps"], guidance=diffusion["guidance"],
        )

        # Independent DDIM (straight from k2_train.main's reference block).
        session = ort.InferenceSession(str(MODEL_DIR / "prior.onnx"),
                                       providers=["CPUExecutionProvider"])
        alphas = np.asarray(diffusion["alphas_cumprod"], dtype=np.float64)
        times = np.linspace(999, 0, diffusion["sample_steps"]).round().astype(int)
        x = noise[None]
        for i, t in enumerate(times):
            a = float(alphas[t])
            a_prev = float(alphas[times[i + 1]]) if i + 1 < len(times) else 1.0
            v_c = session.run(None, {"x": x, "t": np.array([t], np.float32),
                                     "cond": cond[None].astype(np.float32)})[0]
            v_u = session.run(None, {"x": x, "t": np.array([t], np.float32),
                                     "cond": np.zeros((1, cond.size), np.float32)})[0]
            v = v_u + diffusion["guidance"] * (v_c - v_u)
            x0 = np.clip(math.sqrt(a) * x - math.sqrt(1 - a) * v, -6, 6)
            eps = math.sqrt(1 - a) * x + math.sqrt(a) * v
            x = (math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps).astype(np.float32)

        same_cond = float(np.abs(latent - x[0]).max())
        head = float(np.abs(cond[:8] - np.asarray(reference["cond_head"])).max())
        drift = float(np.abs(latent - np.asarray(reference["latent"])).max())
        print(f"\ntext2voxel reference parity ({'placeholder' if meta.get('smoke') else 'trained'}"
              f" model): same-condition latent {same_cond:.3e}; vs training machine: "
              f"cond_head {head:.3e}, latent {drift:.3e}")
        assert np.linalg.norm(cond) == pytest.approx(1.0, abs=1e-5)
        assert same_cond < 1e-4, f"sampler differs from the reference DDIM by {same_cond}"
        assert head < 2e-2, f"cond_head differs from the training machine by {head}"
        assert drift < 0.25, f"latent differs from the training machine by {drift}"

    def test_generates_a_scaled_coloured_glb(self, real_provider, meta, tmp_path) -> None:
        import trimesh

        spec = DesignSpec(object="office chair", dimensions=Dimensions(height_mm=900))
        result = _first_non_empty(real_provider, "a red office chair with armrests and wheels",
                                  tmp_path, spec, steps=12)
        assert result.path.is_file() and result.file_format == "glb"
        data = result.metadata
        assert data["steps"] == 12
        assert data["guidance"] == meta["diffusion"]["guidance"]
        assert data["smoke"] is bool(meta.get("smoke", False))
        assert data["occupied_voxels"] > 0
        assert data["canonical_frame"] is True
        assert set(data["timings_ms"]) == {"load", "encode", "sample", "decode", "mesh", "write"}
        assert data["scale"]["rule"] == "height"

        mesh = trimesh.load(result.path, force="mesh")
        assert mesh.extents[1] == pytest.approx(0.9, abs=1e-4)
        assert mesh.bounds[0][1] == pytest.approx(0.0, abs=1e-4)
        assert getattr(mesh.visual, "kind", None) == "vertex"

    def test_same_seed_same_shape_and_a_seed_is_always_recorded(
        self, real_provider, tmp_path
    ) -> None:
        first = _first_non_empty(real_provider, "a wooden table", tmp_path / "a", steps=6)
        seed = first.metadata["seed"]
        again = real_provider.generate_from_text("a wooden table", tmp_path / "b",
                                                 seed=seed, steps=6)
        assert again.metadata["occupied_voxels"] == first.metadata["occupied_voxels"]
        assert again.metadata["vertices"] == first.metadata["vertices"]

        noise = np.random.default_rng(0).standard_normal(256).astype(np.float32)
        _, one = real_provider.sample_latent("a lamp", noise, steps=4)
        _, two = real_provider.sample_latent("a lamp", noise, steps=4)
        np.testing.assert_array_equal(one, two)

        try:
            unseeded = real_provider.generate_from_text("a wooden table", tmp_path / "c",
                                                        steps=4)
            assert 0 <= unseeded.metadata["seed"] < 2**31
        except GenerationError as exc:   # only an all-zero grid can still be empty
            assert "empty" in exc.user_message

    def test_near_empty_output_is_surfaced_not_failed(self, real_provider, meta,
                                                      tmp_path) -> None:
        """Like the browser: a grid that barely reaches the threshold still gives a mesh."""
        for seed in range(4):
            result = real_provider.generate_from_text("a red office chair", tmp_path / str(seed),
                                                      seed=seed, steps=6)
            data = result.metadata
            assert data["occupied_voxels"] > 0 and data["faces"] > 0
            assert data["iso_level"] <= data["threshold"] == meta["threshold"]
            if data["iso_level"] < data["threshold"]:
                assert data["warnings"][0].startswith("Only ")

    def test_cancellation_stops_between_steps(self, real_provider, tmp_path) -> None:
        messages = []

        def progress(fraction: float, message: str) -> None:
            messages.append(message)
            if message.startswith("Sampling step 3/"):
                raise JobCancelled()

        with pytest.raises(JobCancelled):
            real_provider.generate_from_text("a chair", tmp_path, seed=1, steps=10,
                                             progress=progress)
        assert sum(message.startswith("Sampling step") for message in messages) == 3
        assert not (tmp_path / "model.glb").exists()
        # The lock was released on the way out, so the provider still works.
        assert real_provider.embed("a chair").shape == (384,)

    def test_release_frees_and_reloads(self, tmp_path) -> None:
        provider = Text2VoxelProvider(MODEL_DIR)
        provider.embed("a sofa")
        assert provider._sessions is not None
        provider.release()
        assert provider._sessions is None
        provider.release()  # idempotent
        assert provider.embed("a sofa").shape == (384,)

    def test_reinstalled_files_are_reloaded(self, tmp_path) -> None:
        import shutil

        copy = tmp_path / "model"
        shutil.copytree(MODEL_DIR, copy)
        provider = Text2VoxelProvider(copy)
        provider.embed("a sofa")
        first = provider._sessions
        provider.embed("a sofa")
        assert provider._sessions is first, "unchanged files: sessions are reused"
        stat = (copy / "meta.json").stat()
        os.utime(copy / "meta.json", (stat.st_atime, stat.st_mtime + 10))
        provider.embed("a sofa")
        assert provider._sessions is not first, "a reinstall must not be served stale"
        provider.release()

    def test_full_pipeline_through_the_api(
        self, client: TestClient, project: dict, meta, monkeypatch, tmp_path
    ) -> None:
        provider = Text2VoxelProvider(MODEL_DIR, steps=8)
        monkeypatch.setitem(get_registry()._threed, "text2voxel", provider)
        # Pick a seed that gives this prompt a surface, then run it for real.
        seed = _first_non_empty(provider, project["prompt"], tmp_path).metadata["seed"]

        started = client.post("/api/generate/pipeline", json={
            "project_id": project["id"], "threed_provider": "text2voxel", "seed": seed,
            "export_formats": ["glb", "stl"],
        })
        assert started.status_code == 202
        job = _wait_for_job(client, started.json()["id"], timeout=180)
        assert job["status"] == "completed", job.get("error")
        stages = {stage["key"]: stage for stage in job["stages"]}
        assert stages["images"]["status"] == "skipped"
        assert stages["reconstruction"]["status"] == "completed"
        if meta.get("smoke"):
            assert "Placeholder" in stages["reconstruction"]["detail"]

        fetched = client.get(f"/api/projects/{project['id']}").json()
        assert fetched["images"] == []
        assert fetched["metadata"]["reconstruction"]["seed"] == seed
        assert fetched["metadata"]["reconstruction"]["prompt"] == project["prompt"]
        assert fetched["metadata"]["mesh_processing"]["alignment"]["aligned"] is False
        assert {"glb", "stl"} <= set(fetched["exports"])
        blueprint = client.get(f"/api/projects/{project['id']}/blueprint").json()
        assert set(blueprint["views"]) == {"front", "side", "top"}


# --------------------------------------------------------------------------
# Orchestration with stub providers
# --------------------------------------------------------------------------


class StubTextProvider(TextTo3DGenerator):
    """Writes a box, and remembers what it was asked for."""

    label = "Stub text-to-3D"

    def __init__(self, name: str = "stub_text") -> None:
        self.name = name
        self.calls: list[dict] = []

    def availability(self) -> Availability:
        return Availability.ok()

    def generate_from_text(self, prompt, output_dir, *, spec=None, seed=None,
                           progress=_noop_progress) -> ModelResult:
        import trimesh

        self.calls.append({"prompt": prompt, "seed": seed, "spec": spec})
        for step in range(4):
            progress((step + 1) / 4, f"Sampling step {step + 1}/4")
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.creation.box(extents=(0.4, 0.55, 0.3))
        path = output_dir / "model.glb"
        mesh.export(path)
        return ModelResult(
            path=path, file_format="glb", provider=self.name, generation_time_s=0.01,
            metadata={"seed": seed if seed is not None else 777, "canonical_frame": True},
        )


class StubImageTo3D(ThreeDGenerator):
    """An image-based provider that needs no GPU, for auto-resolution tests."""

    name = "stub_image3d"
    label = "Stub image-to-3D"

    def __init__(self) -> None:
        self.calls = 0

    def availability(self) -> Availability:
        return Availability.ok()

    def generate(self, images: list[ImageResult], output_dir, *, spec=None, resolution=128,
                 progress=_noop_progress) -> ModelResult:
        import trimesh

        self.calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model.glb"
        trimesh.creation.box(extents=(0.3, 0.3, 0.3)).export(path)
        return ModelResult(path=path, file_format="glb", provider=self.name)


class StubImageProvider:
    """Just enough of an ImageGenerator for availability checks."""

    name = "stub_images"
    label = "Stub images"

    def __init__(self, available: bool) -> None:
        self._available = available

    def availability(self) -> Availability:
        return Availability.ok() if self._available else Availability.missing("no images here")

    def release(self) -> None:
        pass


@pytest.fixture
def stub_text(monkeypatch) -> StubTextProvider:
    provider = StubTextProvider()
    monkeypatch.setitem(get_registry()._threed, "stub_text", provider)
    return provider


@pytest.fixture
def stub_fallback(monkeypatch) -> StubTextProvider:
    """A stub installed as *the* text2voxel provider, so "auto" can fall back to it."""
    provider = StubTextProvider(name="text2voxel")
    monkeypatch.setitem(get_registry()._threed, "text2voxel", provider)
    return provider


@pytest.fixture
def stub_image3d(monkeypatch) -> StubImageTo3D:
    import app.providers.registry as registry_module

    provider = StubImageTo3D()
    monkeypatch.setitem(get_registry()._threed, "stub_image3d", provider)
    monkeypatch.setattr(registry_module, "THREED_PREFERENCE", ("stub_image3d",))
    return provider


@pytest.fixture
def image_providers(monkeypatch) -> None:
    registry = get_registry()
    monkeypatch.setitem(registry._image, "stub_up", StubImageProvider(True))
    monkeypatch.setitem(registry._image, "stub_down", StubImageProvider(False))


class TestAutoResolution:
    def test_falls_back_only_without_views_and_without_an_image_provider(
        self, stub_fallback, stub_image3d, image_providers
    ) -> None:
        registry = get_registry()
        pick = registry.threed_provider
        assert pick("auto", allow_text_fallback=True, image_provider="stub_down") is stub_fallback
        assert pick("auto", allow_text_fallback=True, image_provider="stub_up") is stub_image3d
        assert pick("auto", allow_text_fallback=False, image_provider="stub_down") is stub_image3d
        # An explicit choice is never second-guessed.
        assert pick("stub_image3d", allow_text_fallback=True,
                    image_provider="stub_down") is stub_image3d

    def test_no_fallback_when_the_text_model_is_missing(
        self, stub_image3d, image_providers
    ) -> None:
        # conftest points TEXT2VOXEL_PATH at an empty directory.
        registry = get_registry()
        assert not registry.threed_provider("text2voxel").availability().available
        assert registry.threed_provider("auto", allow_text_fallback=True,
                                        image_provider="stub_down") is stub_image3d


class TestTextPath:
    def test_generate_3d_needs_no_views_and_passes_the_raw_prompt(
        self, client: TestClient, project: dict, stub_text
    ) -> None:
        started = client.post("/api/generate/3d", json={
            "project_id": project["id"], "provider": "stub_text", "seed": 42,
        })
        assert started.status_code == 202
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "completed", job.get("error")

        call, = stub_text.calls
        assert call["prompt"] == project["prompt"]
        assert call["seed"] == 42
        expected_height = project["design_spec"]["dimensions"]["height_mm"]
        assert call["spec"].dimensions.height_mm == expected_height

        fetched = client.get(f"/api/projects/{project['id']}").json()
        assert fetched["status"] == "model_ready"
        assert fetched["stats"]["triangles"] == 12
        assert fetched["metadata"]["reconstruction"]["seed"] == 42
        assert "from the prompt text" in fetched["history"][-1]["detail"]

    def test_an_image_provider_still_demands_views(
        self, client: TestClient, project: dict, stub_image3d
    ) -> None:
        started = client.post("/api/generate/3d", json={
            "project_id": project["id"], "provider": "stub_image3d",
        })
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "failed"
        assert "view" in (job["error"] or "").lower()
        assert stub_image3d.calls == 0

    def test_auto_uses_the_views_when_there_are_some(
        self, client: TestClient, project: dict, stub_fallback, stub_image3d, image_providers,
        silhouette_images,
    ) -> None:
        image = silhouette_images[0]
        client.post(
            f"/api/projects/{project['id']}/images/{image.view.value}",
            files={"file": ("front.png", image.path.read_bytes(), "image/png")},
        )
        started = client.post("/api/generate/3d", json={
            "project_id": project["id"], "provider": "auto",
        })
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "completed", job.get("error")
        assert stub_image3d.calls == 1 and stub_fallback.calls == []

    def test_auto_generate_3d_falls_back_to_text_without_views(
        self, client: TestClient, project: dict, stub_fallback, stub_image3d, monkeypatch,
        image_providers,
    ) -> None:
        monkeypatch.setattr(get_registry().settings, "image_provider", "stub_down")
        started = client.post("/api/generate/3d", json={
            "project_id": project["id"], "provider": "auto",
        })
        job = _wait_for_job(client, started.json()["id"])
        assert job["status"] == "completed", job.get("error")
        assert len(stub_fallback.calls) == 1 and stub_image3d.calls == 0


class TestFullPipelineImageSkip:
    def _run(self, client: TestClient, project_id: str, **body) -> dict:
        started = client.post("/api/generate/pipeline", json={"project_id": project_id, **body})
        assert started.status_code == 202, started.text
        return _wait_for_job(client, started.json()["id"], timeout=180)

    def test_explicit_text_provider_skips_the_images_stage(
        self, client: TestClient, project: dict, stub_text
    ) -> None:
        job = self._run(client, project["id"], threed_provider="stub_text", seed=7,
                        export_formats=["glb", "stl"])
        assert job["status"] == "completed", job.get("error")
        stages = {stage["key"]: stage for stage in job["stages"]}
        assert stages["prompt_analysis"]["status"] == "completed"
        assert stages["design_spec"]["status"] == "completed"
        assert stages["images"]["status"] == "skipped"
        assert "prompt text" in stages["images"]["detail"]
        for key in ("reconstruction", "mesh", "export"):
            assert stages[key]["status"] == "completed", key
        assert job["result"]["images"]["skipped"] is True
        assert stub_text.calls[0]["seed"] == 7

        fetched = client.get(f"/api/projects/{project['id']}").json()
        assert fetched["status"] == "mesh_processed"
        assert {"glb", "stl"} <= set(fetched["exports"])
        # The text model's frame is canonical, so mesh processing kept it.
        alignment = fetched["metadata"]["mesh_processing"]["alignment"]
        assert alignment["aligned"] is False

    def test_auto_skips_images_when_no_image_provider_can_run(
        self, client: TestClient, project: dict, stub_fallback, stub_image3d, image_providers
    ) -> None:
        job = self._run(client, project["id"], image_provider="stub_down",
                        threed_provider="auto", export_formats=["glb"])
        assert job["status"] == "completed", job.get("error")
        stages = {stage["key"]: stage for stage in job["stages"]}
        assert stages["images"]["status"] == "skipped"
        assert len(stub_fallback.calls) == 1 and stub_image3d.calls == 0

    def test_without_the_text_model_the_image_stage_still_runs(
        self, client: TestClient, project: dict, stub_image3d, image_providers
    ) -> None:
        # stub_down cannot render, and text2voxel is not installed (conftest):
        # the pipeline must fail at the images stage, exactly as before.
        job = self._run(client, project["id"], image_provider="stub_down",
                        threed_provider="auto")
        assert job["status"] == "failed"
        stages = {stage["key"]: stage for stage in job["stages"]}
        assert stages["images"]["status"] == "failed"
        assert stub_image3d.calls == 0
