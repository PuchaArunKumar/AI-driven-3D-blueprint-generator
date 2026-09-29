"""Install the Text2Voxel-64 model files for the ``text2voxel`` 3D provider.

The model is not committed to the repository (the files are gitignored). This
script reads ``models/manifest.json`` at the repository root - the same
manifest the frontend's ``fetch-models.mjs`` uses - and downloads:

* ``decoder.onnx``, ``prior.onnx``, ``meta.json`` from the GitHub release named
  in the manifest (``text2voxel-v1``), and
* the MiniLM text encoder (``model_quantized.onnx``, ``tokenizer.json``) from
  Hugging Face at a pinned revision,

into ``backend/models/text2voxel`` (``TEXT2VOXEL_PATH``), with the encoder in
its ``minilm/`` subfolder. About 55 MB in total. A file is verified against
the manifest's sha256 when the manifest gives one.

Usage (from the ``backend`` directory)::

    python scripts/install_text2voxel.py            # fetch whatever is missing
    python scripts/install_text2voxel.py --force    # replace every file
    python scripts/install_text2voxel.py --check    # report status only

``--force`` is how the untrained development placeholder (``meta.json`` has
``"smoke": true``) is swapped for the trained release once it is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.logging_conf import configure_logging  # noqa: E402

logger = logging.getLogger("install_text2voxel")

DEFAULT_MANIFEST = REPO_ROOT / "models" / "manifest.json"
CHUNK = 1 << 20


class InstallError(Exception):
    """A download or verification failure with a message fit for the console."""


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InstallError(f"Manifest not found: {path}") from exc
    except ValueError as exc:
        raise InstallError(f"Manifest is not valid JSON ({path}): {exc}") from exc
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise InstallError(f"Manifest lists no files: {path}")
    return manifest


def target_for(key: str, model_dir: Path) -> Path:
    """Map a manifest key onto the backend layout.

    ``text2voxel/<name>`` lands in the model directory itself and anything
    else (``minilm/<name>``) in a subfolder of it. Keys that would escape the
    directory are refused.
    """
    relative = key.removeprefix("text2voxel/")
    target = (model_dir / relative).resolve()
    root = model_dir.resolve()
    if root not in target.parents:
        raise InstallError(f"Refusing a manifest key outside the model directory: {key}")
    return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str | None, release: str) -> str:
    """Stream ``url`` to ``target`` through a sha256; replace only on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "install_text2voxel"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            next_report = 0.25
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                digest.update(block)
                received += len(block)
                if total and received / total >= next_report and received < total:
                    logger.info("  %s: %.1f / %.1f MB", target.name, received / 1e6, total / 1e6)
                    next_report += 0.25
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        if exc.code == 404 and "github.com" in url:
            raise InstallError(
                f"{url} was not found (HTTP 404). Has the '{release}' release been "
                "published with this asset? The model is trained on Kaggle and "
                "released afterwards - see training/text2voxel/README.md."
            ) from exc
        raise InstallError(f"{url} failed: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise InstallError(f"{url} failed: {exc}") from exc

    actual = digest.hexdigest()
    if expected and actual != expected.lower():
        partial.unlink(missing_ok=True)
        raise InstallError(
            f"Checksum mismatch for {target.name}: expected {expected}, got {actual}."
        )
    os.replace(partial, target)
    return actual


def describe_meta(model_dir: Path) -> None:
    try:
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("meta.json is missing or unreadable")
        return
    logger.info("Model: %s v%s (%s)", meta.get("name", "?"), meta.get("version", "?"),
                meta.get("created", "?"))
    if meta.get("smoke"):
        logger.warning(
            "These are the untrained placeholder files (meta.json has \"smoke\": true): "
            "the pipeline runs, but the shapes are meaningless. Re-run with --force "
            "once the trained release is published."
        )


def check(manifest: dict, model_dir: Path) -> int:
    missing = 0
    for key, entry in manifest["files"].items():
        target = target_for(key, model_dir)
        if not target.is_file():
            logger.info("%-30s MISSING", key)
            missing += 1
            continue
        expected = entry.get("sha256")
        if expected:
            state = "ok (sha256 verified)" if sha256_of(target) == expected.lower() \
                else "PRESENT BUT CHECKSUM MISMATCH"
            if "MISMATCH" in state:
                missing += 1
        else:
            state = "present (no checksum in the manifest to verify against)"
        logger.info("%-30s %s", key, state)
    if not missing:
        describe_meta(model_dir)
    return 0 if missing == 0 else 1


def install(manifest: dict, model_dir: Path, force: bool) -> int:
    release = manifest.get("release", "text2voxel")
    failures = 0
    for key, entry in manifest["files"].items():
        target = target_for(key, model_dir)
        expected = entry.get("sha256")
        if target.is_file() and not force:
            if not expected:
                logger.info("%s: present - skipping (no checksum to verify; --force "
                            "replaces it)", key)
                continue
            if sha256_of(target) == expected.lower():
                logger.info("%s: present and verified - skipping", key)
                continue
            logger.info("%s: checksum differs from the manifest - downloading again", key)

        logger.info("%s: downloading %s", key, entry["url"])
        try:
            actual = download(entry["url"], target, expected, release)
        except InstallError as exc:
            logger.error("%s", exc)
            failures += 1
            continue
        logger.info("%s: done (%.1f MB, sha256 %s%s)", key, target.stat().st_size / 1e6,
                    actual[:12], ", verified" if expected else "")

    if failures:
        logger.error("%d file(s) could not be installed into %s", failures, model_dir)
        return 1
    describe_meta(model_dir)
    logger.info("Text2Voxel-64 ready in %s. Pick 'text2voxel' as the 3D provider, or "
                "leave it on auto.", model_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Text2Voxel-64 model files.")
    parser.add_argument("--force", action="store_true",
                        help="download every file again, replacing what is there")
    parser.add_argument("--check", action="store_true", help="report status and exit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"manifest path (default {DEFAULT_MANIFEST})")
    parser.add_argument("--target", type=Path, default=None,
                        help="model directory (default TEXT2VOXEL_PATH, "
                             "backend/models/text2voxel)")
    arguments = parser.parse_args()

    configure_logging("INFO")
    model_dir = arguments.target or Path(get_settings().text2voxel_path)
    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.check:
            return check(manifest, model_dir)
        return install(manifest, model_dir, arguments.force)
    except InstallError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
