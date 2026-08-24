"""Load versioned benchmark semantics without consulting runtime settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from geovision.domain.benchmark import M1ExperimentConfig, TrackerProfile


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return document


def load_m1_experiment_config(path: str | Path) -> M1ExperimentConfig:
    """Resolve tracker profile files and validate one immutable M1 configuration."""

    config_path = Path(path).resolve()
    config_directory = config_path.parent
    document = dict(_read_yaml_mapping(config_path))

    if "trackers" in document:
        raise ValueError("use tracker_profile_files; inline tracker profiles are not supported")
    profile_files = document.pop("tracker_profile_files", None)
    if not isinstance(profile_files, list) or len(profile_files) != 2:
        raise ValueError("tracker_profile_files must contain exactly two relative paths")

    trackers: list[TrackerProfile] = []
    for profile_file in profile_files:
        if not isinstance(profile_file, str) or not profile_file:
            raise ValueError("tracker profile paths must be non-empty strings")
        profile_path = (config_directory / profile_file).resolve()
        if not profile_path.is_relative_to(config_directory):
            raise ValueError("tracker profile paths must remain inside the config directory")
        trackers.append(TrackerProfile.model_validate(_read_yaml_mapping(profile_path)))

    document["trackers"] = trackers
    return M1ExperimentConfig.model_validate(document)


__all__ = ["load_m1_experiment_config"]
