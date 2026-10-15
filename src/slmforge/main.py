"""FastAPI application for SLMForge.

Serves the JSON API under `/api` and, when a built frontend exists, the SPA from
`web/dist`. In development run the API with `slmforge serve` and the Vite dev
server separately; Vite proxies `/api` here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import api_router
from .config import settings
from .db import init_db
from .log import get_logger, setup_logging

logger = get_logger("main")

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    settings.ensure_dirs()
    init_db()
    logger.info(
        "SLMForge %s ready (env=%s, db=%s, sandbox=%s)",
        __version__,
        settings.env,
        settings.database_url,
        settings.sandbox_backend,
    )
    yield
    logger.info("SLMForge shutting down")


def create_app(web_dist: Path | None = None) -> FastAPI:
    """Build the FastAPI application.

    `web_dist` overrides where the built SPA is looked for; it exists so tests can
    exercise both the "assets present" and "API only" code paths.
    """
    app = FastAPI(
        title="SLMForge",
        version=__version__,
        summary="Prompt-to-SLM: build a task-specific small language model from a description.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    _mount_web(app, web_dist or WEB_DIST)
    return app


def _mount_web(app: FastAPI, web_dist: Path = WEB_DIST) -> None:
    """Serve the built SPA when present, with a helpful message when it isn't."""
    index_html = web_dist / "index.html"
    if index_html.is_file():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> Response:
            return FileResponse(index_html)

        @app.get("/{path:path}", include_in_schema=False)
        def spa_fallback(path: str) -> Response:
            # Annotate as Response: a union of response classes is not a valid
            # FastAPI response model and fails at import time.
            root = web_dist.resolve()
            candidate = (root / path).resolve()
            # `path` is attacker-controlled; without this check `../../.env` would
            # escape web_dist and FileResponse would happily serve it.
            if path and root in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index_html)

    else:

        @app.get("/", include_in_schema=False)
        def api_only() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "slmforge",
                    "version": __version__,
                    "message": "API is running. The web UI is not built yet.",
                    "web_dev": "cd web && npm install && npm run dev",
                    "docs": "/docs",
                }
            )


app = create_app()
