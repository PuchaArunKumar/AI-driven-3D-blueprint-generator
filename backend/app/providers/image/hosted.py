"""Hosted image-generation providers (OpenAI DALL-E, Stability AI).

Both read their credentials from the environment and report themselves as
unavailable - with an actionable reason - when the key is missing, rather than
failing at generation time.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import httpx

from app.errors import GenerationError, ProviderUnavailableError
from app.pipeline.prompt_parser import NEGATIVE_PROMPT, build_view_prompt
from app.providers.base import (
    Availability,
    ImageGenerator,
    ImageResult,
    ProgressCallback,
    _noop_progress,
)
from app.schemas import DesignSpec, ViewName

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = httpx.Timeout(180.0, connect=15.0)


def _save(image_bytes: bytes, path: Path) -> tuple[int, int]:
    """Write image bytes to disk and return the decoded (width, height)."""
    from PIL import Image  # noqa: PLC0415

    path.write_bytes(image_bytes)
    with Image.open(path) as opened:
        return opened.width, opened.height


class OpenAIImageProvider(ImageGenerator):
    """DALL-E via the OpenAI Images API."""

    name = "openai"
    label = "DALL-E (OpenAI)"

    #: DALL-E 3 only accepts this fixed set of sizes.
    _DALLE3_SIZES = ("1024x1024", "1792x1024", "1024x1792")

    def __init__(self, api_key: str, model: str = "dall-e-3") -> None:
        self.api_key = api_key
        self.model = model

    def availability(self) -> Availability:
        if not self.api_key:
            return Availability.missing(
                "No OpenAI API key configured.",
                requires=["OPENAI_API_KEY"],
            )
        return Availability.ok()

    def _size_for(self, width: int, height: int) -> str:
        if self.model == "dall-e-3":
            if width > height:
                return "1792x1024"
            if height > width:
                return "1024x1792"
            return "1024x1024"
        # dall-e-2 supports these square sizes.
        for candidate in (256, 512, 1024):
            if width <= candidate:
                return f"{candidate}x{candidate}"
        return "1024x1024"

    def generate(
        self,
        spec: DesignSpec,
        views: list[ViewName],
        output_dir: Path,
        *,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        progress: ProgressCallback = _noop_progress,
    ) -> list[ImageResult]:
        if not self.api_key:
            raise ProviderUnavailableError(
                "Image generation requires an OpenAI API key.",
                hint="Add OPENAI_API_KEY to your .env file, then restart the backend.",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ImageResult] = []
        size = self._size_for(width, height)

        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            for index, view in enumerate(views):
                progress(index / len(views), f"Requesting {view.value} view from DALL-E")
                try:
                    response = client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "prompt": build_view_prompt(spec, view)[:4000],
                            "n": 1,
                            "size": size,
                            "response_format": "b64_json",
                        },
                    )
                except httpx.HTTPError as exc:
                    raise GenerationError(
                        "Could not reach the OpenAI API.",
                        hint="Check your network connection and try again.",
                        detail=str(exc),
                    ) from exc

                if response.status_code == 401:
                    raise ProviderUnavailableError(
                        "The OpenAI API key was rejected.",
                        hint="Verify OPENAI_API_KEY in your .env file.",
                    )
                if response.status_code == 429:
                    raise GenerationError(
                        "OpenAI rate limit reached.",
                        hint="Wait a moment and try again, or reduce the number of views.",
                    )
                if response.status_code >= 400:
                    raise GenerationError(
                        "OpenAI rejected the image request.",
                        hint="Try rephrasing the prompt.",
                        detail=response.text[:2000],
                    )

                payload = response.json()["data"][0]
                image_bytes = base64.b64decode(payload["b64_json"])
                path = output_dir / f"{view.value}.png"
                actual_width, actual_height = _save(image_bytes, path)
                results.append(
                    ImageResult(view=view, path=path, width=actual_width,
                                height=actual_height, provider=self.name)
                )

        progress(1.0, f"Rendered {len(results)} views")
        return results


class StabilityImageProvider(ImageGenerator):
    """Stable Diffusion via the hosted Stability AI REST API."""

    name = "stability"
    label = "Stable Diffusion (Stability AI)"

    def __init__(self, api_key: str, model: str = "stable-image/generate/core") -> None:
        self.api_key = api_key
        self.model = model.strip("/")

    def availability(self) -> Availability:
        if not self.api_key:
            return Availability.missing(
                "No Stability AI API key configured.",
                requires=["STABILITY_API_KEY"],
            )
        return Availability.ok()

    @staticmethod
    def _aspect_ratio(width: int, height: int) -> str:
        ratio = width / height if height else 1.0
        options = {"1:1": 1.0, "3:2": 1.5, "16:9": 1.777, "2:3": 0.667, "9:16": 0.5625}
        return min(options, key=lambda key: abs(options[key] - ratio))

    def generate(
        self,
        spec: DesignSpec,
        views: list[ViewName],
        output_dir: Path,
        *,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        progress: ProgressCallback = _noop_progress,
    ) -> list[ImageResult]:
        if not self.api_key:
            raise ProviderUnavailableError(
                "Image generation requires a Stability AI API key.",
                hint="Add STABILITY_API_KEY to your .env file, then restart the backend.",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        base_seed = seed if seed is not None else int(time.time()) % (2**31 - 1)
        results: list[ImageResult] = []
        url = f"https://api.stability.ai/v2beta/{self.model}"

        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            for index, view in enumerate(views):
                progress(index / len(views), f"Requesting {view.value} view from Stability AI")
                view_seed = (base_seed + index * 1013) % (2**31 - 1)
                try:
                    response = client.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Accept": "image/*",
                        },
                        files={"none": ""},
                        data={
                            "prompt": build_view_prompt(spec, view)[:9000],
                            "negative_prompt": NEGATIVE_PROMPT[:9000],
                            "aspect_ratio": self._aspect_ratio(width, height),
                            "seed": view_seed,
                            "output_format": "png",
                        },
                    )
                except httpx.HTTPError as exc:
                    raise GenerationError(
                        "Could not reach the Stability AI API.",
                        hint="Check your network connection and try again.",
                        detail=str(exc),
                    ) from exc

                if response.status_code in (401, 403):
                    raise ProviderUnavailableError(
                        "The Stability AI API key was rejected.",
                        hint="Verify STABILITY_API_KEY in your .env file.",
                    )
                if response.status_code >= 400:
                    raise GenerationError(
                        "Stability AI rejected the image request.",
                        detail=response.text[:2000],
                    )

                path = output_dir / f"{view.value}.png"
                actual_width, actual_height = _save(response.content, path)
                results.append(
                    ImageResult(view=view, path=path, width=actual_width,
                                height=actual_height, provider=self.name, seed=view_seed)
                )

        progress(1.0, f"Rendered {len(results)} views")
        return results
