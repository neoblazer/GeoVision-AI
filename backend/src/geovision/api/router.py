"""Versioned HTTP routes."""

from fastapi import APIRouter

from geovision import __version__
from geovision.core.config import Settings


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": __version__}

    @router.get("/config", tags=["system"])
    def public_config() -> dict[str, object]:
        return {
            "environment": settings.environment,
            "detector_backend": settings.detector_backend,
            "detector_model": settings.detector_model,
            "detector_candidate_model": settings.detector_candidate_model,
            "tracker_backend": settings.tracker_backend,
            "tracker_reid": settings.tracker_reid,
            "depth_enabled": settings.depth_enabled,
            "segmentation_enabled": settings.segmentation_enabled,
        }

    return router

