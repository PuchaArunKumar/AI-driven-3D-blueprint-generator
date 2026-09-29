"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import generate, projects, system
from app.api import jobs as jobs_api
from app.config import get_settings
from app.db import init_db
from app.demo_catalog import adopt_demo_projects
from app.errors import BlueprintError
from app.jobs import manager
from app.logging_conf import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
Turn a natural-language product description into a design specification,
multi-view concept images, a 3D model, an engineering blueprint and
downloadable CAD assets.

The pipeline is **Text -> Image -> 3D -> GLB -> BLEND**, or **Text -> 3D**
directly with the Text2Voxel-64 provider. Every provider reports its own
availability; nothing is simulated.
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.storage_dir / "logs")
    settings.ensure_directories()
    init_db()
    # The demo assets are committed but their database rows are not; register
    # any that are missing so a fresh clone opens onto a populated Gallery.
    try:
        adopt_demo_projects()
    except Exception:  # pragma: no cover - never block startup on the demos
        logger.warning("Could not register the demo projects", exc_info=True)
    manager.bind_loop(asyncio.get_running_loop())
    logger.info("AI-Driven 3D Blueprint Generator %s ready", __version__)
    logger.info("Storage: %s", settings.storage_dir)
    try:
        yield
    finally:
        manager.shutdown()
        try:
            from app.providers.registry import get_registry  # noqa: PLC0415

            get_registry().release_all()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("Provider cleanup failed", exc_info=True)
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI-Driven 3D Blueprint Generator",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------- handlers
    @app.exception_handler(BlueprintError)
    async def handle_domain_error(request: Request, exc: BlueprintError) -> JSONResponse:
        """Return the curated message; the traceback stays in the log."""
        logger.warning("%s %s -> %s: %s", request.method, request.url.path,
                       exc.code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request,
                                exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", [])[1:]) or "request"
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": f"{field}: {first.get('msg', 'is invalid')}",
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "Something went wrong on the server.",
                "hint": "Check the backend log for details.",
            },
        )

    # -------------------------------------------------------------- routing
    app.include_router(projects.router)
    app.include_router(generate.router)
    app.include_router(jobs_api.router)
    app.include_router(system.router)

    @app.get("/api/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    # Generated artefacts are served read-only from the storage root.
    app.mount("/files", StaticFiles(directory=settings.storage_dir), name="files")
    return app


app = create_app()
