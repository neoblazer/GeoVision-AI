"""Immutable contracts for the Milestone 1 perception benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class BenchmarkContract(BaseModel):
    """Strict base model for persisted benchmark configuration and metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    strict_scalar_types: ClassVar[dict[str, type[object]]] = {}

    @model_validator(mode="before")
    @classmethod
    def reject_scalar_type_coercion(cls, value: object) -> object:
        if isinstance(value, Mapping):
            for field_name, expected_type in cls.strict_scalar_types.items():
                if (
                    field_name in value
                    and value[field_name] is not None
                    and type(value[field_name]) is not expected_type
                ):
                    raise ValueError(
                        f"{field_name} must have exact type {expected_type.__name__}"
                    )
        return value


class SourceType(StrEnum):
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"


class ReadinessStatus(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"


class SourceProfile(BenchmarkContract):
    strict_scalar_types = {"comparative": bool}

    source_id: str = Field(min_length=1)
    source_type: SourceType
    comparative: bool

    @model_validator(mode="after")
    def validate_comparative_source(self) -> SourceProfile:
        if self.comparative and self.source_type != SourceType.FILE:
            raise ValueError("comparative matrix sources must be recorded files")
        return self


class DetectorProfile(BenchmarkContract):
    strict_scalar_types = {"weights_size_bytes": int}

    profile_id: Literal["yolo11n", "yolo26n"]
    backend: Literal["ultralytics"]
    model: Literal["yolo11n.pt", "yolo26n.pt"]
    weights_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    weights_size_bytes: int = Field(gt=0)


class DetectorComparisonPolicy(BenchmarkContract):
    """Shared detector semantics applied identically to both M1 checkpoints."""

    strict_scalar_types = {
        "image_size": int,
        "confidence": float,
        "iou": float,
        "max_det": int,
        "rect": bool,
        "agnostic_nms": bool,
        "augment": bool,
        "compile": bool,
        "channels_last": bool,
    }

    task: Literal["detect"]
    image_size: Literal[640]
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    iou: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    max_det: Literal[300]
    rect: Literal[True]
    classes: None
    agnostic_nms: Literal[False]
    augment: Literal[False]
    end_to_end: Literal["model-native"]
    compile: Literal[False]
    channels_last: Literal[False]


class TrackerProfile(BenchmarkContract):
    strict_scalar_types = {
        "track_high_thresh": float,
        "track_low_thresh": float,
        "new_track_thresh": float,
        "track_buffer": int,
        "match_thresh": float,
        "fuse_score": bool,
        "reid": bool,
        "proximity_thresh": float,
        "appearance_thresh": float,
    }

    profile_id: str = Field(min_length=1)
    backend: Literal["bytetrack", "botsort"]
    track_high_thresh: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    track_low_thresh: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    new_track_thresh: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    track_buffer: int = Field(gt=0)
    track_buffer_unit: Literal["frames"]
    match_thresh: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fuse_score: bool
    gmc_method: Literal["sparseOptFlow"] | None = None
    reid: Literal[False]
    proximity_thresh: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    appearance_thresh: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    model: Literal["auto"] | None = None

    @model_validator(mode="after")
    def validate_m1_tracker_policy(self) -> TrackerProfile:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh cannot exceed track_high_thresh")
        if self.backend == "bytetrack":
            if self.profile_id != "bytetrack":
                raise ValueError("ByteTrack must use the bytetrack profile ID")
            if self.gmc_method is not None:
                raise ValueError("Milestone 1 ByteTrack does not use native GMC")
            if any(
                value is not None
                for value in (self.proximity_thresh, self.appearance_thresh, self.model)
            ):
                raise ValueError("ByteTrack cannot contain BoT-SORT-only semantics")
        else:
            if self.profile_id != "botsort-sparse-optical-flow":
                raise ValueError("BoT-SORT must use the botsort-sparse-optical-flow profile ID")
            if self.gmc_method != "sparseOptFlow":
                raise ValueError("Milestone 1 BoT-SORT requires native sparseOptFlow GMC")
            if self.proximity_thresh is None or self.appearance_thresh is None:
                raise ValueError("BoT-SORT requires proximity and appearance thresholds")
            if self.model != "auto":
                raise ValueError("Milestone 1 BoT-SORT requires model=auto with ReID disabled")
        return self


class ResolvedCombination(BenchmarkContract):
    combination_id: str = Field(min_length=1)
    detector: DetectorProfile
    tracker: TrackerProfile


class MeasurementPolicy(BenchmarkContract):
    strict_scalar_types = {
        "warmup_frames": int,
        "fresh_detector_pass_required_for_publishable_run": bool,
        "same_run_resume_cache_allowed": bool,
        "cross_run_cache_allowed_for_development": bool,
        "cross_run_cache_publishable": bool,
    }

    warmup_frames: Literal[30]
    fresh_detector_pass_required_for_publishable_run: Literal[True] = True
    same_run_resume_cache_allowed: Literal[True] = True
    cross_run_cache_allowed_for_development: Literal[True] = True
    cross_run_cache_publishable: Literal[False] = False


class ArtifactSchemaIdentifiers(BenchmarkContract):
    """Schema IDs, with experiment manifest as the sole run-manifest artifact."""

    frame_metadata: Literal["geovision.frame-metadata/v1"]
    detection_cache: Literal["geovision.detection-cache/v1"]
    track_artifact: Literal["geovision.track-artifact/v1"]
    runtime_telemetry: Literal["geovision.runtime-telemetry/v1"]
    experiment_manifest: Literal["geovision.experiment-manifest/v1"]
    failure_report: Literal["geovision.failure-report/v1"]
    completion_marker: Literal["geovision.completion-marker/v1"]
    mot_sidecar: Literal["geovision.mot-sidecar/v1"]
    detection_cache_format: Literal["jsonl"]
    manifest_format: Literal["json"]


class RecordedSourcePolicy(BenchmarkContract):
    """Fail-closed source semantics for future comparative recorded-file execution."""

    strict_scalar_types = {
        "frame_index_origin": int,
        "contiguous_frame_indices": bool,
        "require_finite_positive_fps": bool,
        "invalid_fps_fails_before_inference": bool,
        "sequential_decode": bool,
        "allow_seek": bool,
        "allow_skip": bool,
        "allow_retry": bool,
        "require_reported_frame_count_match": bool,
        "channel_count": int,
        "source_paths_in_artifacts": bool,
    }

    frame_index_origin: Literal[0]
    contiguous_frame_indices: Literal[True]
    canonical_timestamp: Literal["frame-index-over-validated-fps"]
    source_timestamp_role: Literal["diagnostic"]
    source_timestamp_statuses: tuple[
        Literal["available"],
        Literal["unavailable"],
        Literal["duplicated"],
        Literal["regressive"],
    ]
    require_finite_positive_fps: Literal[True]
    invalid_fps_fails_before_inference: Literal[True]
    sequential_decode: Literal[True]
    allow_seek: Literal[False]
    allow_skip: Literal[False]
    allow_retry: Literal[False]
    require_reported_frame_count_match: Literal[True]
    early_decode_failure: Literal["fail-run"]
    ambiguous_eof_without_frame_count: Literal["fail-run"]
    pixel_format: Literal["BGR"]
    pixel_dtype: Literal["uint8"]
    channel_layout: Literal["HWC"]
    channel_count: Literal[3]
    dimension_authority: Literal["decoded-frame"]
    source_fingerprint: Literal["sha256-and-byte-size"]
    replay_frame_fingerprint: Literal["sha256"]
    source_paths_in_artifacts: Literal[False]


class CanonicalArtifactPolicy(BenchmarkContract):
    """Deterministic ordering and JSON rules that participate in cache identity."""

    strict_scalar_types = {
        "preserve_full_finite_precision": bool,
        "normalize_negative_zero": bool,
        "sort_keys": bool,
        "compact_separators": bool,
        "allow_nan": bool,
    }

    detection_order: tuple[
        Literal["confidence-desc"],
        Literal["class-id-asc"],
        Literal["x1-asc"],
        Literal["y1-asc"],
        Literal["x2-asc"],
        Literal["y2-asc"],
        Literal["original-ordinal-asc"],
    ]
    external_scalar_conversion: Literal["python-numeric"]
    float_representation: Literal["python-3.11-shortest-round-trip"]
    preserve_full_finite_precision: Literal[True]
    normalize_negative_zero: Literal[True]
    encoding: Literal["utf-8"]
    sort_keys: Literal[True]
    compact_separators: Literal[True]
    allow_nan: Literal[False]


class DeferredServices(BenchmarkContract):
    strict_scalar_types = {
        "camera_motion_service": bool,
        "depth": bool,
        "distance": bool,
        "segmentation": bool,
        "mission_memory": bool,
        "events": bool,
        "dashboard": bool,
        "replay": bool,
        "scheduling": bool,
        "llm_reporting": bool,
    }

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
    strict_scalar_types = {
        "device": int,
        "batch_size": int,
        "fp16": bool,
        "automatic_dependency_install": bool,
        "automatic_model_download": bool,
        "hardware_confirmed": bool,
    }

    python_version: Literal["3.11"]
    ultralytics_version: str | None = Field(default=None, min_length=1)
    torch_version: str | None = Field(default=None, min_length=1)
    cuda_version: str | None = Field(default=None, min_length=1)
    device: Literal[0]
    batch_size: Literal[1]
    fp16: Literal[True]
    automatic_dependency_install: Literal[False]
    automatic_model_download: Literal[False]
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
    detector_comparison: DetectorComparisonPolicy
    trackers: tuple[TrackerProfile, ...] = Field(min_length=2, max_length=2)
    runtime: RuntimeProfile
    measurement: MeasurementPolicy
    artifacts: ArtifactSchemaIdentifiers
    recorded_source: RecordedSourcePolicy
    canonical_artifacts: CanonicalArtifactPolicy
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
        detector_by_id = {detector.profile_id: detector for detector in self.detectors}
        expected_detectors = {
            "yolo11n": (
                "yolo11n.pt",
                5_613_764,
            ),
            "yolo26n": (
                "yolo26n.pt",
                5_544_453,
            ),
        }
        if set(detector_by_id) != set(expected_detectors):
            raise ValueError("Milestone 1 requires YOLO11n and YOLO26n profiles")
        for profile_id, (model, size_bytes) in expected_detectors.items():
            detector = detector_by_id[profile_id]
            if detector.model != model or detector.weights_size_bytes != size_bytes:
                raise ValueError(f"{profile_id} model and checkpoint byte size must match")

        tracker_ids = [tracker.profile_id for tracker in self.trackers]
        if len(tracker_ids) != len(set(tracker_ids)):
            raise ValueError("tracker profile IDs must be unique")
        if {tracker.backend for tracker in self.trackers} != {"bytetrack", "botsort"}:
            raise ValueError("Milestone 1 requires ByteTrack and BoT-SORT")
        expected_tracker_arguments = (0.25, 0.10, 0.25, 30, 0.80, True)
        for tracker in self.trackers:
            if self.detector_comparison.confidence > tracker.track_low_thresh:
                raise ValueError("detector confidence cannot exceed tracker low threshold")
            actual_arguments = (
                tracker.track_high_thresh,
                tracker.track_low_thresh,
                tracker.new_track_thresh,
                tracker.track_buffer,
                tracker.match_thresh,
                tracker.fuse_score,
            )
            if actual_arguments != expected_tracker_arguments:
                raise ValueError("Milestone 1 tracker arguments must match the locked profile")

        botsort = next(tracker for tracker in self.trackers if tracker.backend == "botsort")
        if botsort.proximity_thresh != 0.50 or botsort.appearance_thresh != 0.80:
            raise ValueError("Milestone 1 BoT-SORT thresholds must match the locked profile")
        if self.detector_comparison.confidence != 0.10:
            raise ValueError("Milestone 1 detector-cache confidence must be 0.10")
        if self.detector_comparison.iou != 0.70:
            raise ValueError("Milestone 1 detector IoU must be 0.70")
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
            _normalize_negative_zero(self.model_dump(mode="json", exclude_none=True)),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _normalize_negative_zero(value: object) -> object:
    """Recursively normalize finite negative zero before canonical JSON serialization."""

    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, dict):
        return {key: _normalize_negative_zero(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_negative_zero(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_negative_zero(item) for item in value)
    return value


__all__ = [
    "ArtifactSchemaIdentifiers",
    "CanonicalArtifactPolicy",
    "DeferredServices",
    "DetectorComparisonPolicy",
    "DetectorProfile",
    "ExperimentReadiness",
    "M1ExperimentConfig",
    "MeasurementPolicy",
    "ReadinessStatus",
    "RecordedSourcePolicy",
    "ResolvedCombination",
    "RuntimeProfile",
    "SourceProfile",
    "SourceType",
    "TrackerProfile",
]
