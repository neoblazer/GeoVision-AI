"""Application configuration with explicit environment-variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Runtime settings kept independent of any web framework."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "GeoVision AI"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    log_level: str = "INFO"

    detector_backend: str = "mock"
    detector_model: str = "yolo11n.pt"
    detector_candidate_model: str = "yolo26n.pt"
    detector_confidence: float = Field(default=0.35, ge=0.0, le=1.0)

    tracker_backend: str = "botsort"
    tracker_reid: bool = False
    camera_motion_backend: str = "sparse_optical_flow"

    depth_enabled: bool = False
    segmentation_enabled: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load the small supported environment surface without hidden magic."""

        def env(name: str, default: str) -> str:
            return os.getenv(f"GEOVISION_{name}", default)

        def env_bool(name: str, default: bool) -> bool:
            value = env(name, str(default)).strip().lower()
            if value in {"1", "true", "yes", "on"}:
                return True
            if value in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"GEOVISION_{name} must be a boolean value")

        return cls(
            environment=env("ENVIRONMENT", "development"),
            log_level=env("LOG_LEVEL", "INFO"),
            detector_backend=env("DETECTOR_BACKEND", "mock"),
            detector_model=env("DETECTOR_MODEL", "yolo11n.pt"),
            tracker_backend=env("TRACKER_BACKEND", "botsort"),
            tracker_reid=env_bool("TRACKER_REID", False),
            depth_enabled=env_bool("DEPTH_ENABLED", False),
            segmentation_enabled=env_bool("SEGMENTATION_ENABLED", False),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings object per process."""

    return Settings.from_environment()

