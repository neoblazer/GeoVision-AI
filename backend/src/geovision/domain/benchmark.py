"""Immutable contracts for the Milestone 1 perception benchmark."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class BenchmarkContract(BaseModel):
    """Strict base model for persisted benchmark configuration and metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceType(StrEnum):
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"


class ReadinessStatus(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"


class SourceProfile(BenchmarkContract):
    source_id: str = Field(min_length=1)
    source_type: SourceType
    comparative: bool

    @model_validator(mode="after")
    def validate_comparative_source(self) -> SourceProfile:
        if self.comparative and self.source_type != SourceType.FILE:
            raise ValueError("comparative matrix sources must be recorded files")
        return self


class DetectorProfile(BenchmarkContract):
    profile_id: str = Field(min_length=1)
    backend: Literal["ultralytics"]
    model: Literal["yolo11n.pt", "yolo26n.pt"]
    weights_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    image_size: int = Field(ge=32)
    confidence: float = Field(ge=0.0, le=1.0)


class TrackerProfile(BenchmarkContract):
    profile_id: str = Field(min_length=1)
    backend: Literal["bytetrack", "botsort"]
    gmc_method: Literal["none", "sparseOptFlow"]
    reid: bool

    @model_validator(mode="after")
    def validate_m1_tracker_policy(self) -> TrackerProfile:
        if self.reid:
            raise ValueError("ReID is disabled in Milestone 1")
        if self.backend == "botsort" and self.gmc_method != "sparseOptFlow":
            raise ValueError("Milestone 1 BoT-SORT requires native sparseOptFlow GMC")
        if self.backend == "bytetrack" and self.gmc_method != "none":
            raise ValueError("Milestone 1 ByteTrack does not use native GMC")
        return self


class ResolvedCombination(BenchmarkContract):
    combination_id: str = Field(min_length=1)
    detector: DetectorProfile
    tracker: TrackerProfile


class MeasurementPolicy(BenchmarkContract):
    warmup_frames: int = Field(default=30, ge=0)
    fresh_detector_pass_required_for_publishable_run: Literal[True] = True
    same_run_resume_cache_allowed: Literal[True] = True
    cross_run_cache_allowed_for_development: Literal[True] = True
    cross_run_cache_publishable: Literal[False] = False


class ArtifactSchemaIdentifiers(BenchmarkContract):
    detection_cache: Literal["geovision.detection-cache/v1"]
    experiment_manifest: Literal["geovision.experiment-manifest/v1"]
    mot_sidecar: Literal["geovision.mot-sidecar/v1"]
    detection_cache_format: Literal["jsonl"]
    manifest_format: Literal["json"]


class DeferredServices(BenchmarkContract):
    camera_motion_service: Literal[False]
    depth: Literal[False]
    distance: Literal[False]
    segmentation: Literal[False]
    mission_memory: Literal[False]
    events: Literal[False]
    dashboard: Literal[False]
    replay: Literal[False]
    scheduling: Literal[False]
    llm_reporting: Literal[False]


class RuntimeProfile(BenchmarkContract):
    python_version: Literal["3.11"]
    ultralytics_version: str | None = Field(default=None, min_length=1)
    torch_version: str | None = Field(default=None, min_length=1)
    cuda_version: str | None = Field(default=None, min_length=1)
    device: str = Field(min_length=1)
    batch_size: int = Field(ge=1)
    fp16: bool
    hardware_confirmed: bool


class ExperimentReadiness(BenchmarkContract):
    status: ReadinessStatus
    unresolved_fields: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == ReadinessStatus.READY

    def require_ready(self) -> None:
        if not self.ready:
            unresolved = ", ".join(self.unresolved_fields)
            raise ValueError(f"experiment is not execution-ready; unresolved: {unresolved}")


class M1ExperimentConfig(BenchmarkContract):
    schema_version: Literal["geovision.m1-experiment/v1"]
    experiment_id: str = Field(min_length=1)
    sources: tuple[SourceProfile, ...] = Field(min_length=1)
    detectors: tuple[DetectorProfile, ...] = Field(min_length=2, max_length=2)
    trackers: tuple[TrackerProfile, ...] = Field(min_length=2, max_length=2)
    runtime: RuntimeProfile
    measurement: MeasurementPolicy
    artifacts: ArtifactSchemaIdentifiers
    services: DeferredServices

    @model_validator(mode="after")
    def validate_m1_matrix(self) -> M1ExperimentConfig:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source profile IDs must be unique")
        if not any(source.comparative for source in self.sources):
            raise ValueError("Milestone 1 requires at least one comparative recorded-file source")

        detector_ids = [detector.profile_id for detector in self.detectors]
        if len(detector_ids) != len(set(detector_ids)):
            raise ValueError("detector profile IDs must be unique")
        if {detector.model for detector in self.detectors} != {"yolo11n.pt", "yolo26n.pt"}:
            raise ValueError("Milestone 1 requires YOLO11n and YOLO26n")
        detector_settings = {
            (detector.image_size, detector.confidence) for detector in self.detectors
        }
        if len(detector_settings) != 1:
            raise ValueError("Milestone 1 detector comparison settings must match")

        tracker_ids = [tracker.profile_id for tracker in self.trackers]
        if len(tracker_ids) != len(set(tracker_ids)):
            raise ValueError("tracker profile IDs must be unique")
        if {tracker.backend for tracker in self.trackers} != {"bytetrack", "botsort"}:
            raise ValueError("Milestone 1 requires ByteTrack and BoT-SORT")
        return self

    def resolve_combinations(self) -> tuple[ResolvedCombination, ...]:
        """Return the fixed four-way detector/tracker matrix in research order."""

        detector_by_model = {detector.model: detector for detector in self.detectors}
        tracker_by_backend = {tracker.backend: tracker for tracker in self.trackers}
        combinations = tuple(
            ResolvedCombination(
                combination_id=f"{detector.profile_id}-{tracker.profile_id}",
                detector=detector,
                tracker=tracker,
            )
            for detector in (
                detector_by_model["yolo11n.pt"],
                detector_by_model["yolo26n.pt"],
            )
            for tracker in (
                tracker_by_backend["bytetrack"],
                tracker_by_backend["botsort"],
            )
        )
        if len({item.combination_id for item in combinations}) != 4:
            raise ValueError("Milestone 1 must resolve to four unique combinations")
        return combinations

    def execution_readiness(self) -> ExperimentReadiness:
        """Report unresolved provenance without preventing draft inspection."""

        unresolved: list[str] = []
        for field_name in ("ultralytics_version", "torch_version", "cuda_version"):
            if getattr(self.runtime, field_name) is None:
                unresolved.append(f"runtime.{field_name}")
        if not self.runtime.hardware_confirmed:
            unresolved.append("runtime.hardware_confirmed")

        detector_by_model = {detector.model: detector for detector in self.detectors}
        for model_name in ("yolo11n.pt", "yolo26n.pt"):
            detector = detector_by_model[model_name]
            if detector.weights_sha256 is None:
                unresolved.append(f"detectors.{detector.profile_id}.weights_sha256")

        status = ReadinessStatus.BLOCKED if unresolved else ReadinessStatus.READY
        return ExperimentReadiness(status=status, unresolved_fields=tuple(unresolved))

    def require_execution_ready(self) -> None:
        """Reject execution until required versions, hashes, and hardware are confirmed."""

        self.execution_readiness().require_ready()

    def canonical_json(self) -> str:
        """Serialize resolved experiment semantics deterministically."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


__all__ = [
    "ArtifactSchemaIdentifiers",
    "DeferredServices",
    "DetectorProfile",
    "ExperimentReadiness",
    "M1ExperimentConfig",
    "MeasurementPolicy",
    "ReadinessStatus",
    "ResolvedCombination",
    "RuntimeProfile",
    "SourceProfile",
    "SourceType",
    "TrackerProfile",
]
