"""Hosted image-to-3D provider.

Posts a reference image to an external image-to-3D inference endpoint and
stores the returned GLB.  This is how a learned reconstruction model such as
TRELLIS/SLAT, or any comparable hosted service, plugs into the pipeline
without the application depending on it.

The endpoint contract is deliberately small so it can front a self-hosted
TRELLIS deployment, a Replicate/fal.ai model, or an in-house service:

    POST  <endpoint>
    multipart/form-data: image=<png bytes>
    ->  either a binary GLB body (Content-Type: model/gltf-binary)
        or JSON {"model_url": "https://..."} / {"model_b64": "<base64 glb>"}

Marked experimental: it is only as available as the endpoint behind it.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import httpx

from app.errors import GenerationError, ProviderUnavailableError
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

REQUEST_TIMEOUT = httpx.Timeout(600.0, connect=15.0)

#: Preference order when choosing the single image to send.
PREFERRED_VIEWS = (ViewName.PERSPECTIVE, ViewName.FRONT, ViewName.RIGHT, ViewName.LEFT)


class HostedThreeDProvider(ThreeDGenerator):
    """Image-to-3D through an external inference endpoint (TRELLIS/SLAT-style)."""

    name = "trellis_api"
    label = "Image-to-3D API (TRELLIS / SLAT-compatible)"

    def __init__(self, endpoint: str, api_key: str = "", timeout: int = 600) -> None:
        self.endpoint = endpoint.strip()
        self.api_key = api_key
        self.timeout = timeout

    def availability(self) -> Availability:
        if not self.endpoint:
            return Availability.missing(
                "No image-to-3D endpoint is configured.",
                requires=["TRELLIS_ENDPOINT"],
                experimental=True,
            )
        return Availability.ok(experimental=True)

    @staticmethod
    def _pick_image(images: list[ImageResult]) -> ImageResult:
        by_view = {image.view: image for image in images}
        for view in PREFERRED_VIEWS:
            if view in by_view:
                return by_view[view]
        return images[0]

    def generate(
        self,
        images: list[ImageResult],
        output_dir: Path,
        *,
        spec: DesignSpec | None = None,
        resolution: int = 128,
        progress: ProgressCallback = _noop_progress,
    ) -> ModelResult:
        if not self.endpoint:
            raise ProviderUnavailableError(
                "3D generation requires a configured inference provider.",
                hint="Set TRELLIS_ENDPOINT in .env, or switch the 3D provider to "
                     "'visual_hull' in Settings to reconstruct locally.",
            )
        if not images:
            raise GenerationError("No images are available to reconstruct from.")

        output_dir.mkdir(parents=True, exist_ok=True)
        chosen = self._pick_image(images)
        started = time.time()
        progress(0.1, f"Uploading {chosen.view.value} view to the inference endpoint")

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.post(
                    self.endpoint,
                    headers=headers,
                    files={"image": (chosen.path.name, chosen.path.read_bytes(), "image/png")},
                )
                progress(0.7, "Reconstruction returned, downloading model")

                if response.status_code in (401, 403):
                    raise ProviderUnavailableError(
                        "The image-to-3D endpoint rejected the credentials.",
                        hint="Check TRELLIS_API_KEY in your .env file.",
                    )
                if response.status_code >= 400:
                    raise GenerationError(
                        "The image-to-3D endpoint returned an error.",
                        detail=response.text[:2000],
                    )

                content_type = response.headers.get("content-type", "")
                if "json" in content_type:
                    payload = response.json()
                    if payload.get("model_b64"):
                        data = base64.b64decode(payload["model_b64"])
                    elif payload.get("model_url"):
                        downloaded = client.get(payload["model_url"])
                        downloaded.raise_for_status()
                        data = downloaded.content
                    else:
                        raise GenerationError(
                            "The endpoint returned JSON without a model.",
                            detail=str(payload)[:1000],
                        )
                else:
                    data = response.content
        except httpx.HTTPError as exc:
            raise GenerationError(
                "Could not reach the image-to-3D endpoint.",
                hint="Check TRELLIS_ENDPOINT and that the service is running.",
                detail=str(exc),
            ) from exc

        if not data:
            raise GenerationError("The image-to-3D endpoint returned an empty model.")

        model_path = output_dir / "model.glb"
        model_path.write_bytes(data)
        elapsed = time.time() - started
        progress(1.0, f"Received model in {elapsed:.1f}s")
        return ModelResult(
            path=model_path,
            file_format="glb",
            provider=self.name,
            generation_time_s=elapsed,
            metadata={
                "method": "hosted image-to-3D inference",
                "endpoint": self.endpoint,
                "source_view": chosen.view.value,
            },
        )
