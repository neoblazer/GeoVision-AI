"""Validated data contracts for the perception and mission pipelines."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from geovision.domain.enums import (
    AssociationOutcome,
    DistanceSource,
    EntityState,
    EventType,
    TrackState,
)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameRef(DomainModel):
    source_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)


class BoundingBox(DomainModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_order(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def footpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


class Detection(DomainModel):
    detection_id: str = Field(min_length=1)
    frame: FrameRef
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class TrackObservation(DomainModel):
    track_id: int = Field(ge=0)
    frame: FrameRef
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    state: TrackState = TrackState.ACTIVE


class MotionEstimate(DomainModel):
    frame: FrameRef
    homography: tuple[float, float, float, float, float, float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    inlier_ratio: float = Field(ge=0.0, le=1.0)
    residual_px: float = Field(ge=0.0)
    reliable: bool


class DistanceEstimate(DomainModel):
    track_id: int = Field(ge=0)
    frame: FrameRef
    distance_m: float | None = Field(default=None, gt=0.0)
    source: DistanceSource = DistanceSource.UNAVAILABLE
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated: bool = False
    warning: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> DistanceEstimate:
        if self.source == DistanceSource.UNAVAILABLE and self.distance_m is not None:
            raise ValueError("unavailable distance cannot contain a metric value")
        return self


class MissionEntity(DomainModel):
    entity_id: str = Field(min_length=1)
    track_ids: tuple[int, ...]
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    first_seen_s: float = Field(ge=0.0)
    last_seen_s: float = Field(ge=0.0)
    observation_count: int = Field(ge=1)
    state: EntityState = EntityState.ACTIVE
    last_known_footpoint: tuple[float, float]

    @model_validator(mode="after")
    def validate_timeline(self) -> MissionEntity:
        if self.last_seen_s < self.first_seen_s:
            raise ValueError("last_seen_s cannot precede first_seen_s")
        if not self.track_ids:
            raise ValueError("mission entity must contain at least one track id")
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("mission entity track ids must be unique")
        return self


class AssociationDecision(DomainModel):
    candidate_track_id: int = Field(ge=0)
    entity_id: str = Field(min_length=1)
    outcome: AssociationOutcome
    total_score: float = Field(ge=0.0, le=1.0)
    motion_score: float = Field(ge=0.0, le=1.0)
    appearance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    scale_score: float = Field(ge=0.0, le=1.0)
    time_score: float = Field(ge=0.0, le=1.0)
    motion_reliability: float = Field(ge=0.0, le=1.0)
    appearance_reliability: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...]


class MissionEvent(DomainModel):
    event_id: str = Field(min_length=1)
    event_type: EventType
    entity_id: str = Field(min_length=1)
    started_at_s: float = Field(ge=0.0)
    confirmed_at_s: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rule_id: str = Field(min_length=1)
    evidence_frame_ids: tuple[int, ...]

    @model_validator(mode="after")
    def validate_event_timeline(self) -> MissionEvent:
        if self.confirmed_at_s < self.started_at_s:
            raise ValueError("confirmed_at_s cannot precede started_at_s")
        if not self.evidence_frame_ids:
            raise ValueError("mission event requires at least one evidence frame")
        return self


class PipelineResult(DomainModel):
    frame: FrameRef
    detections: tuple[Detection, ...]
    tracks: tuple[TrackObservation, ...]

    @model_validator(mode="after")
    def validate_frame_consistency(self) -> PipelineResult:
        if any(detection.frame != self.frame for detection in self.detections):
            raise ValueError("every detection must reference the PipelineResult frame")
        if any(track.frame != self.frame for track in self.tracks):
            raise ValueError("every track must reference the PipelineResult frame")
        return self
