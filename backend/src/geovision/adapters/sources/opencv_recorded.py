"""Strict one-shot OpenCV adapter for comparative recorded-file sources."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

import cv2
import numpy as np

from geovision.domain.benchmark import RecordedSourcePolicy
from geovision.domain.m1_artifacts import (
    RecordedSourceMetadata,
    SourceFingerprint,
    SourceTimestampStatus,
    UnavailableMetadataField,
    UnavailableMetadataReason,
    UnavailableSourceMetadata,
)
from geovision.domain.models import FrameRef
from geovision.ports.sources import (
    DecodedFrame,
    RecordedFrameIterator,
    SourceDecodeError,
    SourceFailureCode,
    SourceLifecycleError,
    SourceValidationError,
)

HASH_CHUNK_SIZE = 1024 * 1024


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> object: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...

    def getBackendName(self) -> object: ...


class OpenCVFacade(Protocol):
    __version__: str
    CAP_PROP_FPS: int
    CAP_PROP_FRAME_COUNT: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_POS_MSEC: int
    CAP_PROP_FOURCC: int

    def VideoCapture(self, filename: str) -> Capture: ...


CaptureFactory = Callable[[str], Capture]
FileIdentity = tuple[int, int, int, int, int]


def _single_path_is_symlink_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _is_symlink_or_reparse(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return any(
        _single_path_is_symlink_or_reparse(component)
        for component in reversed((absolute, *absolute.parents))
    )


def _stat_identity(metadata: os.stat_result) -> FileIdentity:
    return (
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_dev,
        metadata.st_ino,
    )


def _stream_sha256(path: Path, *, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _OpenCVFrameIterator:
    """Eagerly owned one-shot iterator with explicit deterministic cleanup."""

    def __init__(self, source: OpenCVRecordedSource) -> None:
        self._source = source
        self._next_frame_id = 0
        self._closed = False
        self._completed = False

    def __iter__(self) -> _OpenCVFrameIterator:
        return self

    def __next__(self) -> DecodedFrame:
        if self._closed:
            if self._completed:
                raise StopIteration
            raise SourceLifecycleError(
                SourceFailureCode.CLOSED,
                "recorded source iterator is closed",
            )

        try:
            expected_count = self._source.metadata.validated_reported_frame_count
            if self._next_frame_id < expected_count:
                success, frame = self._source._read_once()
                if not success:
                    raise SourceDecodeError(
                        SourceFailureCode.EARLY_DECODE_FAILURE,
                        "decode failed before the reported frame count",
                    )
                decoded = self._source._decoded_frame(self._next_frame_id, frame)
                self._next_frame_id += 1
                return decoded

            extra_success, _ = self._source._read_once()
            if extra_success:
                raise SourceDecodeError(
                    SourceFailureCode.FRAME_COUNT_MISMATCH,
                    "decoded data exceeds the reported frame count",
                )
            self._source._require_stable_source_identity(during_decode=True)
        except BaseException:
            self.close()
            raise

        self._completed = True
        self.close()
        raise StopIteration

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._source.close()

    def _mark_source_closed(self) -> None:
        self._closed = True

    def __enter__(self) -> _OpenCVFrameIterator:
        if self._closed:
            raise SourceLifecycleError(
                SourceFailureCode.CLOSED,
                "recorded source iterator is closed",
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class OpenCVRecordedSource:
    """Synchronous, unbuffered source enforcing locked M1 recorded-file semantics."""

    def __init__(
        self,
        source_path: str | Path,
        source_id: str,
        policy: RecordedSourcePolicy,
        *,
        capture_factory: CaptureFactory | None = None,
        opencv_facade: OpenCVFacade | None = None,
    ) -> None:
        if not isinstance(policy, RecordedSourcePolicy):
            raise TypeError("policy must be a RecordedSourcePolicy")

        self._policy = policy
        self._capture: Capture | None = None
        self._closed = False
        self._released = False
        self._iteration_started = False
        self._active_iterator: _OpenCVFrameIterator | None = None
        self._last_valid_source_timestamp_ms: float | None = None

        resolved_path, source_fingerprint, source_identity = self._fingerprint_source(
            source_path,
            source_id,
        )
        self._source_path = resolved_path
        self._source_identity = source_identity
        facade = opencv_facade or cast(OpenCVFacade, cv2)
        self._opencv_facade = facade
        factory = capture_factory or facade.VideoCapture

        try:
            self._require_stable_source_identity(during_decode=False)
            try:
                self._capture = factory(str(resolved_path))
                opened = self._capture.isOpened()
            except Exception as exc:
                raise SourceValidationError(
                    SourceFailureCode.CAPTURE_OPEN_FAILED,
                    "capture could not be initialized",
                ) from exc
            if type(opened) is not bool or not opened:
                raise SourceValidationError(
                    SourceFailureCode.CAPTURE_OPEN_FAILED,
                    "capture did not open",
                )
            self._require_stable_source_identity(during_decode=False)
            self._metadata = self._read_source_metadata(
                source_fingerprint=source_fingerprint,
                facade=facade,
            )
        except BaseException:
            self._close_after_initialization_failure()
            raise

    @property
    def metadata(self) -> RecordedSourceMetadata:
        return self._metadata

    @staticmethod
    def _fingerprint_source(
        source_path: str | Path,
        source_id: str,
    ) -> tuple[Path, SourceFingerprint, FileIdentity]:
        candidate = Path(source_path)
        if _is_symlink_or_reparse(candidate):
            raise SourceValidationError(
                SourceFailureCode.SYMLINK_FORBIDDEN,
                "source indirection is forbidden",
            )
        if not candidate.exists():
            raise SourceValidationError(SourceFailureCode.MISSING, "source file is missing")
        if not candidate.is_file():
            raise SourceValidationError(
                SourceFailureCode.NOT_REGULAR_FILE,
                "source must be a regular file",
            )

        try:
            resolved = candidate.resolve(strict=True)
            before = resolved.stat()
            digest = _stream_sha256(resolved, chunk_size=HASH_CHUNK_SIZE)
            after = resolved.stat()
        except FileNotFoundError as exc:
            raise SourceValidationError(
                SourceFailureCode.CHANGED_DURING_HASH,
                "source changed during fingerprinting",
            ) from exc
        except OSError as exc:
            raise SourceValidationError(
                SourceFailureCode.NOT_REGULAR_FILE,
                "source could not be read as a regular file",
            ) from exc

        if _stat_identity(before) != _stat_identity(after):
            raise SourceValidationError(
                SourceFailureCode.CHANGED_DURING_HASH,
                "source changed during fingerprinting",
            )
        identity = _stat_identity(after)
        return (
            resolved,
            SourceFingerprint(
                source_id=source_id,
                sha256=digest,
                byte_size=after.st_size,
            ),
            identity,
        )

    def _require_stable_source_identity(self, *, during_decode: bool) -> None:
        error_type = SourceDecodeError if during_decode else SourceValidationError
        try:
            current_identity = _stat_identity(self._source_path.stat())
        except OSError as exc:
            raise error_type(
                SourceFailureCode.SOURCE_CHANGED,
                "source identity changed",
            ) from exc
        if current_identity != self._source_identity:
            raise error_type(
                SourceFailureCode.SOURCE_CHANGED,
                "source identity changed",
            )

    def _read_source_metadata(
        self,
        *,
        source_fingerprint: SourceFingerprint,
        facade: OpenCVFacade,
    ) -> RecordedSourceMetadata:
        fps = self._required_fps(self._capture_property(facade.CAP_PROP_FPS))
        frame_count = self._required_frame_count(
            self._capture_property(facade.CAP_PROP_FRAME_COUNT)
        )
        width_value, width_read_reason = self._optional_capture_property(
            facade.CAP_PROP_FRAME_WIDTH
        )
        height_value, height_read_reason = self._optional_capture_property(
            facade.CAP_PROP_FRAME_HEIGHT
        )
        width, width_reason = self._optional_dimension(width_value, width_read_reason)
        height, height_reason = self._optional_dimension(height_value, height_read_reason)
        backend, backend_reason = self._capture_backend()
        fourcc, fourcc_reason = self._capture_fourcc(facade)

        unavailable = tuple(
            UnavailableSourceMetadata(field=field, reason=reason)
            for field, value, reason in (
                (UnavailableMetadataField.REPORTED_WIDTH, width, width_reason),
                (UnavailableMetadataField.REPORTED_HEIGHT, height, height_reason),
                (UnavailableMetadataField.CAPTURE_BACKEND, backend, backend_reason),
                (UnavailableMetadataField.CODEC_FOURCC, fourcc, fourcc_reason),
            )
            if value is None and reason is not None
        )
        return RecordedSourceMetadata(
            source=source_fingerprint,
            reported_width=width,
            reported_height=height,
            validated_fps=fps,
            validated_reported_frame_count=frame_count,
            opencv_version=facade.__version__,
            capture_backend=backend,
            codec_fourcc=fourcc,
            unavailable_metadata=unavailable,
        )

    def _capture_property(self, property_id: int) -> object:
        capture = self._require_capture()
        try:
            return capture.get(property_id)
        except Exception as exc:
            raise SourceValidationError(
                SourceFailureCode.CAPTURE_OPEN_FAILED,
                "capture metadata could not be read",
            ) from exc

    def _optional_capture_property(
        self,
        property_id: int,
    ) -> tuple[object | None, UnavailableMetadataReason | None]:
        capture = self._require_capture()
        try:
            return capture.get(property_id), None
        except Exception:
            return None, UnavailableMetadataReason.UNSUPPORTED

    def _require_capture(self) -> Capture:
        if self._capture is None:
            raise SourceLifecycleError(
                SourceFailureCode.CLOSED,
                "recorded source has no active capture",
            )
        return self._capture

    @staticmethod
    def _required_fps(value: object) -> float:
        number = _strict_number(value)
        if number is None or number <= 0.0:
            raise SourceValidationError(
                SourceFailureCode.INVALID_FPS,
                "reported FPS must be a finite positive number",
            )
        return number

    @staticmethod
    def _required_frame_count(value: object) -> int:
        number = _strict_number(value)
        if number is None or number < 0.0 or not number.is_integer():
            raise SourceValidationError(
                SourceFailureCode.INVALID_FRAME_COUNT,
                "reported frame count must be a finite non-negative integer",
            )
        return int(number)

    @staticmethod
    def _optional_dimension(
        value: object,
        read_reason: UnavailableMetadataReason | None = None,
    ) -> tuple[int | None, UnavailableMetadataReason | None]:
        if read_reason is not None:
            return None, read_reason
        if value is None or (type(value) in {int, float} and float(value) == 0.0):
            return None, UnavailableMetadataReason.NOT_REPORTED
        number = _strict_number(value)
        if number is None or number <= 0.0 or not number.is_integer():
            return None, UnavailableMetadataReason.INVALID
        return int(number), None

    def _capture_backend(self) -> tuple[str | None, UnavailableMetadataReason | None]:
        capture = self._require_capture()
        get_backend_name = getattr(capture, "getBackendName", None)
        if not callable(get_backend_name):
            return None, UnavailableMetadataReason.UNSUPPORTED
        try:
            backend = get_backend_name()
        except Exception:
            return None, UnavailableMetadataReason.UNSUPPORTED
        if backend is None or backend == "":
            return None, UnavailableMetadataReason.NOT_REPORTED
        if type(backend) is not str:
            return None, UnavailableMetadataReason.INVALID
        return backend, None

    def _capture_fourcc(
        self,
        facade: OpenCVFacade,
    ) -> tuple[str | None, UnavailableMetadataReason | None]:
        try:
            property_id = facade.CAP_PROP_FOURCC
        except AttributeError:
            return None, UnavailableMetadataReason.UNSUPPORTED
        raw_value, read_reason = self._optional_capture_property(property_id)
        if read_reason is not None:
            return None, read_reason
        if raw_value is None or (
            type(raw_value) in {int, float} and float(raw_value) == 0.0
        ):
            return None, UnavailableMetadataReason.NOT_REPORTED
        number = _strict_number(raw_value)
        if number is None or number < 0.0 or not number.is_integer():
            return None, UnavailableMetadataReason.INVALID
        code = int(number)
        if code > 0xFFFFFFFF:
            return None, UnavailableMetadataReason.INVALID
        characters = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
        if len(characters) != 4 or any(
            not 32 <= ord(character) <= 126 for character in characters
        ):
            return None, UnavailableMetadataReason.INVALID
        return characters, None

    def frames(self) -> RecordedFrameIterator:
        if self._iteration_started:
            raise SourceLifecycleError(
                SourceFailureCode.ALREADY_CONSUMED,
                "recorded source iteration is one-shot",
            )
        if self._closed:
            raise SourceLifecycleError(SourceFailureCode.CLOSED, "recorded source is closed")
        self._iteration_started = True
        iterator = _OpenCVFrameIterator(self)
        self._active_iterator = iterator
        return iterator

    def _read_once(self) -> tuple[bool, object]:
        capture = self._require_capture()
        try:
            result = capture.read()
        except Exception as exc:
            raise SourceDecodeError(
                SourceFailureCode.CAPTURE_READ_FAILED,
                "capture read raised an exception",
            ) from exc
        if type(result) is not tuple or len(result) != 2 or type(result[0]) is not bool:
            raise SourceDecodeError(
                SourceFailureCode.CAPTURE_READ_FAILED,
                "capture read returned an invalid result",
            )
        return result

    def _decoded_frame(self, frame_id: int, frame: object) -> DecodedFrame:
        if not isinstance(frame, np.ndarray):
            raise SourceDecodeError(
                SourceFailureCode.INVALID_FRAME,
                "decoded frame must be a NumPy ndarray",
            )
        if frame.dtype != np.dtype(np.uint8):
            raise SourceDecodeError(
                SourceFailureCode.INVALID_FRAME,
                "decoded frame dtype must be uint8",
            )
        if frame.ndim != 3 or frame.shape[2] != self._policy.channel_count:
            raise SourceDecodeError(
                SourceFailureCode.INVALID_FRAME,
                "decoded frame shape must be H x W x 3",
            )
        height, width, _ = frame.shape
        if height <= 0 or width <= 0:
            raise SourceDecodeError(
                SourceFailureCode.INVALID_FRAME,
                "decoded frame dimensions must be positive",
            )
        source_timestamp_ms, timestamp_status = self._diagnostic_source_timestamp()
        frame_ref = FrameRef(
            source_id=self._metadata.source.source_id,
            frame_id=self._policy.frame_index_origin + frame_id,
            timestamp_s=frame_id / self._metadata.validated_fps,
        )
        try:
            return DecodedFrame.from_capture(
                frame,
                frame_ref=frame_ref,
                source_timestamp_ms=source_timestamp_ms,
                source_timestamp_status=timestamp_status,
            )
        except (TypeError, ValueError) as exc:
            raise SourceDecodeError(
                SourceFailureCode.INVALID_FRAME,
                "decoded frame could not establish immutable owned storage",
            ) from exc

    def _diagnostic_source_timestamp(
        self,
    ) -> tuple[float | None, SourceTimestampStatus]:
        """Compare against the last strictly increasing available timestamp."""

        capture = self._require_capture()
        try:
            raw_value = capture.get(self._opencv_facade.CAP_PROP_POS_MSEC)
        except Exception:
            return None, SourceTimestampStatus.UNAVAILABLE
        number = _strict_number(raw_value)
        if number is None or number < 0.0:
            return None, SourceTimestampStatus.UNAVAILABLE
        previous = self._last_valid_source_timestamp_ms
        if previous is None or number > previous:
            self._last_valid_source_timestamp_ms = number
            return number, SourceTimestampStatus.AVAILABLE
        if number == previous:
            return number, SourceTimestampStatus.DUPLICATED
        return number, SourceTimestampStatus.REGRESSIVE

    def _close_after_initialization_failure(self) -> None:
        if self._capture is None or self._released:
            return
        self._released = True
        self._closed = True
        try:
            self._capture.release()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_iterator is not None:
            self._active_iterator._mark_source_closed()
        if self._capture is None or self._released:
            return
        self._released = True
        try:
            self._capture.release()
        except Exception as exc:
            raise SourceLifecycleError(
                SourceFailureCode.RELEASE_FAILED,
                "capture release failed",
            ) from exc

    def __enter__(self) -> OpenCVRecordedSource:
        if self._closed:
            raise SourceLifecycleError(SourceFailureCode.CLOSED, "recorded source is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["OpenCVRecordedSource"]
