"""FastAPI application entry point."""

from fastapi import FastAPI

from geovision.api.router import create_router
from geovision.core.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    application = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        description="Persistent, explainable mission intelligence for UAV video.",
    )
    application.include_router(create_router(resolved), prefix=resolved.api_prefix)
    return application


app = create_app()

