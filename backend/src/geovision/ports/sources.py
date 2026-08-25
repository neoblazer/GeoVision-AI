"""Framework-independent operational boundary for recorded frame sources."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from geovision.domain.m1_artifacts import (
    FrameMetadata,
    RecordedSourceMetadata,
    SourceTimestampStatus,
)
from geovision.domain.models import FrameRef


class SourceFailureCode(StrEnum):
    MISSING = "missing"
    NOT_REGULAR_FILE = "not_regular_file"
    SYMLINK_FORBIDDEN = "symlink_forbidden"
    CHANGED_DURING_HASH = "changed_during_hash"
    CAPTURE_OPEN_FAILED = "capture_open_failed"
    INVALID_FPS = "invalid_fps"
    INVALID_FRAME_COUNT = "invalid_frame_count"
    CAPTURE_READ_FAILED = "capture_read_failed"
    EARLY_DECODE_FAILURE = "early_decode_failure"
    FRAME_COUNT_MISMATCH = "frame_count_mismatch"
    INVALID_FRAME = "invalid_frame"
    ALREADY_CONSUMED = "already_consumed"
    CLOSED = "closed"
    RELEASE_FAILED = "release_failed"
    SOURCE_CHANGED = "source_changed"


class RecordedSourceError(RuntimeError):
    """Base error carrying an artifact-safe source failure category."""

    def __init__(self, code: SourceFailureCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class SourceValidationError(RecordedSourceError):
    """A source path, fingerprint, capture, or metadata contract is invalid."""


class SourceDecodeError(RecordedSourceError):
    """Sequential decoding violated the locked recorded-source contract."""


class SourceLifecycleError(RecordedSourceError):
    """A one-shot source was used after consumption or closure."""


def _sha256_bgr(bgr: NDArray[np.uint8]) -> str:
    return hashlib.sha256(memoryview(bgr).cast("B")).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class DecodedFrame:
    """One decoded frame with adapter-owned immutable pixel storage.

    Diagnostic source time and status never change canonical
    ``FrameRef.timestamp_s``; canonical time remains
    ``frame_id / validated_fps``.
    """

    frame_metadata: FrameMetadata
    bgr: NDArray[np.uint8]

    def __init__(self) -> None:
        raise TypeError("use DecodedFrame.from_capture()")

    @classmethod
    def from_capture(
        cls,
        bgr: object,
        *,
        frame_ref: FrameRef,
        source_timestamp_ms: float | None,
        source_timestamp_status: SourceTimestampStatus,
    ) -> DecodedFrame:
        """Copy capture storage so backend buffer reuse cannot change frame identity."""

        if not isinstance(bgr, np.ndarray):
            raise TypeError("bgr must be a NumPy ndarray")
        if bgr.dtype != np.dtype(np.uint8):
            raise ValueError("bgr dtype must be exactly uint8")
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr shape must be exactly H x W x 3")
        height, width, _ = bgr.shape
        if height <= 0 or width <= 0:
            raise ValueError("bgr dimensions must be positive")

        owned = np.array(bgr, dtype=np.uint8, order="C", copy=True, subok=False)
        if not owned.flags.c_contiguous or not owned.flags.owndata or owned.base is not None:
            raise ValueError("owned bgr storage could not be established")
        metadata = FrameMetadata(
            frame=frame_ref,
            decoded_width=width,
            decoded_height=height,
            source_timestamp_ms=source_timestamp_ms,
            source_timestamp_status=source_timestamp_status,
            decoded_bgr_sha256=_sha256_bgr(owned),
        )
        owned.setflags(write=False)

        instance = object.__new__(cls)
        object.__setattr__(instance, "frame_metadata", metadata)
        object.__setattr__(instance, "bgr", owned)
        return instance


class RecordedFrameIterator(Iterator[DecodedFrame], Protocol):
    def close(self) -> None: ...

    def __enter__(self) -> RecordedFrameIterator: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class RecordedFrameSource(Protocol):
    """One-shot recorded source with canonical and diagnostic time separation.

    Diagnostic timestamps are compared with the last strictly increasing
    available diagnostic timestamp. The first finite non-negative value is
    available and establishes that baseline. A higher value is available and
    replaces it, an equal value is duplicated, and a lower value is regressive.
    Unavailable and regressive samples do not replace or reset the baseline.
    Diagnostic status never changes canonical ``FrameRef.timestamp_s``, which
    remains ``frame_id / validated_fps``.
    """

    @property
    def metadata(self) -> RecordedSourceMetadata: ...

    def frames(self) -> RecordedFrameIterator: ...

    def close(self) -> None: ...

    def __enter__(self) -> RecordedFrameSource: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


__all__ = [
    "DecodedFrame",
    "RecordedFrameIterator",
    "RecordedFrameSource",
    "RecordedSourceError",
    "SourceDecodeError",
    "SourceFailureCode",
    "SourceLifecycleError",
    "SourceValidationError",
]
