"""Serializable, path-free artifacts for Milestone 1 recorded-file execution."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from geovision.domain.models import FrameRef

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ArtifactContract(BaseModel):
    """Strict immutable base for deterministic Pydantic artifact serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceTimestampStatus(StrEnum):
    """Classify diagnostic time against the last increasing available baseline.

    The first finite non-negative diagnostic timestamp is ``available`` and
    establishes the baseline. A value above that baseline is ``available`` and
    becomes the new baseline, an equal value is ``duplicated``, and a lower
    value is ``regressive``. An ``unavailable`` or ``regressive`` sample does
    not replace or reset the baseline. Diagnostic status never changes the
    canonical ``FrameRef.timestamp_s``, which remains
    ``frame_id / validated_fps``.
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DUPLICATED = "duplicated"
    REGRESSIVE = "regressive"


class UnavailableMetadataField(StrEnum):
    REPORTED_WIDTH = "reported_width"
    REPORTED_HEIGHT = "reported_height"
    CAPTURE_BACKEND = "capture_backend"
    CODEC_FOURCC = "codec_fourcc"


class UnavailableMetadataReason(StrEnum):
    NOT_REPORTED = "not_reported"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


class UnavailableSourceMetadata(ArtifactContract):
    field: UnavailableMetadataField
    reason: UnavailableMetadataReason


class SourceFingerprint(ArtifactContract):
    source_id: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)


class RecordedSourceMetadata(ArtifactContract):
    source: SourceFingerprint
    reported_width: int | None = Field(default=None, gt=0)
    reported_height: int | None = Field(default=None, gt=0)
    validated_fps: FiniteFloat = Field(gt=0.0)
    validated_reported_frame_count: int = Field(ge=0)
    opencv_version: str = Field(min_length=1)
    capture_backend: str | None = Field(default=None, min_length=1)
    codec_fourcc: str | None = Field(default=None, min_length=4, max_length=4)
    unavailable_metadata: tuple[UnavailableSourceMetadata, ...]

    @field_validator("codec_fourcc")
    @classmethod
    def require_printable_ascii_fourcc(cls, value: str | None) -> str | None:
        if value is not None and any(not 32 <= ord(character) <= 126 for character in value):
            raise ValueError("codec_fourcc must contain exactly four printable ASCII characters")
        return value

    @model_validator(mode="after")
    def validate_unavailable_metadata(self) -> RecordedSourceMetadata:
        optional_fields = (
            UnavailableMetadataField.REPORTED_WIDTH,
            UnavailableMetadataField.REPORTED_HEIGHT,
            UnavailableMetadataField.CAPTURE_BACKEND,
            UnavailableMetadataField.CODEC_FOURCC,
        )
        expected = tuple(
            field_name
            for field_name in optional_fields
            if getattr(self, field_name.value) is None
        )
        actual = tuple(item.field for item in self.unavailable_metadata)
        if len(actual) != len(set(actual)):
            raise ValueError("unavailable metadata fields must be unique")
        if actual != expected:
            raise ValueError(
                "unavailable metadata must exactly describe absent fields in canonical order"
            )
        return self


class FrameMetadata(ArtifactContract):
    frame: FrameRef
    decoded_width: int = Field(gt=0)
    decoded_height: int = Field(gt=0)
    source_timestamp_ms: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Diagnostic source timestamp compared against the last strictly increasing "
            "available diagnostic timestamp; it never replaces canonical FrameRef.timestamp_s."
        ),
    )
    source_timestamp_status: SourceTimestampStatus = Field(
        description=(
            "The first finite non-negative diagnostic value is available and establishes the "
            "baseline. Values above, equal to, or below the baseline are available, duplicated, "
            "or regressive respectively. Unavailable and regressive samples do not replace or "
            "reset the baseline. Canonical time remains frame_id / validated_fps."
        )
    )
    decoded_bgr_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("frame", mode="before")
    @classmethod
    def reject_coercive_frame_scalars(cls, value: object) -> object:
        if isinstance(value, Mapping):
            frame_id = value.get("frame_id")
            timestamp_s = value.get("timestamp_s")
            if "frame_id" in value and type(frame_id) is not int:
                raise ValueError("frame_id must be an exact integer")
            if "timestamp_s" in value and (
                type(timestamp_s) not in {int, float} or not isfinite(float(timestamp_s))
            ):
                raise ValueError("timestamp_s must be a finite number")
        return value

    @model_validator(mode="after")
    def validate_source_timestamp(self) -> FrameMetadata:
        if self.source_timestamp_status == SourceTimestampStatus.UNAVAILABLE:
            if self.source_timestamp_ms is not None:
                raise ValueError("unavailable source timestamp cannot contain a value")
        elif self.source_timestamp_ms is None:
            raise ValueError("available source timestamp status requires a value")
        return self


class CheckpointFingerprint(ArtifactContract):
    detector_profile_id: str = Field(min_length=1)
    expected_filename: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)

    @field_validator("expected_filename")
    @classmethod
    def require_filename_only(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("expected_filename must contain a filename only")
        return value


__all__ = [
    "CheckpointFingerprint",
    "FrameMetadata",
    "RecordedSourceMetadata",
    "SourceFingerprint",
    "SourceTimestampStatus",
    "UnavailableMetadataField",
    "UnavailableMetadataReason",
    "UnavailableSourceMetadata",
]
