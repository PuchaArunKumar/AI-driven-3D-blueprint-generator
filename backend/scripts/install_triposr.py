"""Install the TripoSR source and pre-fetch its weights.

TripoSR is MIT-licensed but is not published on PyPI and has no setup.py, so
it cannot be pip-installed. It is deliberately not vendored into this
repository either - this script fetches it into ``backend/vendor/TripoSR``
(gitignored) so the upstream project stays upstream.

Usage (from the ``backend`` directory)::

    python scripts/install_triposr.py            # clone + fetch weights
    python scripts/install_triposr.py --check    # report status only

The heavyweight CUDA extension TripoSR normally needs (``torchmcubes``) is
*not* installed: the provider substitutes scikit-image's marching cubes, so no
build toolchain is required.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.logging_conf import configure_logging  # noqa: E402

logger = logging.getLogger("install_triposr")

REPOSITORY = "https://github.com/VAST-AI-Research/TripoSR.git"
MODEL_ID = "stabilityai/TripoSR"
EXTRA_PACKAGES = ("omegaconf", "einops")


def clone(target: Path) -> bool:
    if (target / "tsr" / "system.py").is_file():
        logger.info("TripoSR source already present at %s", target)
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", REPOSITORY, target)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", REPOSITORY, str(target)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        logger.error("git clone failed: %s", (completed.stderr or "").strip()[-500:])
        return False
    return (target / "tsr" / "system.py").is_file()


def install_packages() -> bool:
    missing = []
    for package in EXTRA_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if not missing:
        return True

    logger.info("Installing %s", ", ".join(missing))
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", *missing],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        logger.error("pip install failed: %s", (completed.stderr or "").strip()[-500:])
        return False
    return True


def fetch_weights() -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub is not installed")
        return False

    for filename in ("config.yaml", "model.ckpt"):
        logger.info("Fetching %s (this is ~1.7 GB on first run)", filename)
        try:
            hf_hub_download(MODEL_ID, filename)
        except Exception as exc:
            logger.error("Could not download %s: %s", filename, exc)
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Install TripoSR.")
    parser.add_argument("--check", action="store_true", help="report status and exit")
    arguments = parser.parse_args()

    configure_logging("INFO")
    settings = get_settings()
    target = Path(settings.triposr_path)

    if arguments.check:
        source_ok = (target / "tsr" / "system.py").is_file()
        logger.info("source at %s: %s", target, "present" if source_ok else "MISSING")
        for package in EXTRA_PACKAGES:
            try:
                __import__(package)
                logger.info("%s: installed", package)
            except ImportError:
                logger.info("%s: MISSING", package)
        return 0 if source_ok else 1

    if not install_packages():
        return 1
    if not clone(target):
        logger.error("Could not obtain the TripoSR source.")
        return 1
    if not fetch_weights():
        logger.error("Source installed, but the weights could not be fetched. "
                     "Re-run this script to retry; downloads resume.")
        return 1

    logger.info("TripoSR ready. Set THREED_PROVIDER=triposr, or pick it in the Studio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
