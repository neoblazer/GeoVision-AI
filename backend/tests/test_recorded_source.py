from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from geovision.adapters.sources import opencv_recorded
from geovision.adapters.sources.opencv_recorded import OpenCVRecordedSource
from geovision.core.experiment_config import load_m1_experiment_config
from geovision.domain.m1_artifacts import (
    FrameMetadata,
    SourceFingerprint,
    SourceTimestampStatus,
    UnavailableMetadataField,
)
from geovision.domain.models import FrameRef
from geovision.ports import sources as source_ports
from geovision.ports.sources import (
    DecodedFrame,
    SourceDecodeError,
    SourceFailureCode,
    SourceLifecycleError,
    SourceValidationError,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "m1-benchmark.yaml"


def fourcc(value: str) -> float:
    return float(sum(ord(character) << (8 * index) for index, character in enumerate(value)))


class FakeOpenCV:
    __version__ = "4.14.0-test"
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_MSEC = 5
    CAP_PROP_FOURCC = 6


class FakeCapture:
    def __init__(
        self,
        responses: list[tuple[bool, object]],
        *,
        opened: bool = True,
        fps: object = 2.0,
        frame_count: object = 0.0,
        width: object = 3.0,
        height: object = 2.0,
        timestamps_ms: list[object] | None = None,
        backend: object = "FAKE",
        codec: object = fourcc("MJPG"),
        property_errors: set[int] | None = None,
        backend_error: Exception | None = None,
        on_read: Callable[[int], None] | None = None,
    ) -> None:
        self.responses = responses
        self.opened = opened
        self.properties = {
            FakeOpenCV.CAP_PROP_FPS: fps,
            FakeOpenCV.CAP_PROP_FRAME_COUNT: frame_count,
            FakeOpenCV.CAP_PROP_FRAME_WIDTH: width,
            FakeOpenCV.CAP_PROP_FRAME_HEIGHT: height,
            FakeOpenCV.CAP_PROP_FOURCC: codec,
        }
        self.timestamps_ms = timestamps_ms or []
        self.backend = backend
        self.property_errors = property_errors or set()
        self.backend_error = backend_error
        self.on_read = on_read
        self.read_count = 0
        self.release_count = 0
        self.timestamp_read_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> object:
        if property_id in self.property_errors:
            raise OSError("controlled capture metadata failure")
        if property_id == FakeOpenCV.CAP_PROP_POS_MSEC:
            index = self.timestamp_read_count
            self.timestamp_read_count += 1
            if index < len(self.timestamps_ms):
                return self.timestamps_ms[index]
            return -1.0
        return self.properties[property_id]

    def read(self) -> tuple[bool, object]:
        index = self.read_count
        self.read_count += 1
        if self.on_read is not None:
            self.on_read(index)
        if index < len(self.responses):
            return self.responses[index]
        return False, None

    def release(self) -> None:
        self.release_count += 1

    def getBackendName(self) -> object:
        if self.backend_error is not None:
            raise self.backend_error
        return self.backend


class ReusingBufferCapture(FakeCapture):
    def __init__(self, payloads: list[np.ndarray]) -> None:
        self.payloads = payloads
        self.shared_buffer = np.empty_like(payloads[0])
        super().__init__(
            [],
            frame_count=float(len(payloads)),
            timestamps_ms=[float(index) for index in range(len(payloads))],
        )

    def read(self) -> tuple[bool, object]:
        index = self.read_count
        self.read_count += 1
        if index >= len(self.payloads):
            return False, None
        self.shared_buffer[...] = self.payloads[index]
        return True, self.shared_buffer


class RecordedSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source_path = self.temporary_root / "source.dat"
        self.source_path.write_bytes(b"stable fake recorded source")
        self.policy = load_m1_experiment_config(CONFIG_PATH).recorded_source

    @staticmethod
    def frame(height: int = 2, width: int = 3) -> np.ndarray:
        return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)

    def source(
        self,
        capture: FakeCapture,
        *,
        source_path: Path | None = None,
    ) -> OpenCVRecordedSource:
        return OpenCVRecordedSource(
            source_path or self.source_path,
            "recorded-file",
            self.policy,
            capture_factory=lambda _: capture,
            opencv_facade=FakeOpenCV(),
        )

    def test_stable_fingerprint_and_path_free_deterministic_metadata(self) -> None:
        first_capture = FakeCapture([], frame_count=0.0)
        first = self.source(first_capture)
        first_json = first.metadata.model_dump_json()
        first.close()

        second_capture = FakeCapture([], frame_count=0.0)
        second = self.source(second_capture)
        second_json = second.metadata.model_dump_json()
        second.close()

        expected_hash = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self.assertEqual(first.metadata.source.sha256, expected_hash)
        self.assertEqual(first.metadata.source.byte_size, self.source_path.stat().st_size)
        self.assertEqual(first_json, second_json)
        self.assertNotIn(str(self.source_path), first_json)
        self.assertNotIn("path", json.loads(first_json)["source"])

    def test_changed_source_bytes_produce_a_different_fingerprint(self) -> None:
        first = self.source(FakeCapture([], frame_count=0.0))
        first_hash = first.metadata.source.sha256
        first.close()
        self.source_path.write_bytes(b"different fake recorded source")

        second = self.source(FakeCapture([], frame_count=0.0))
        self.addCleanup(second.close)

        self.assertNotEqual(first_hash, second.metadata.source.sha256)

    def test_source_change_during_hash_is_rejected(self) -> None:
        original_stream_sha256 = opencv_recorded._stream_sha256

        def changing_hash(path: Path, *, chunk_size: int) -> str:
            digest = original_stream_sha256(path, chunk_size=chunk_size)
            with path.open("ab") as source:
                source.write(b"changed")
            return digest

        with patch.object(opencv_recorded, "_stream_sha256", changing_hash):
            with self.assertRaises(SourceValidationError) as raised:
                self.source(FakeCapture([], frame_count=0.0))

        self.assertEqual(raised.exception.code, SourceFailureCode.CHANGED_DURING_HASH)

    def test_source_change_during_capture_construction_releases_once(self) -> None:
        capture = FakeCapture([], frame_count=0.0)

        def changing_factory(_: str) -> FakeCapture:
            self.source_path.write_bytes(b"changed during capture construction")
            return capture

        with self.assertRaises(SourceValidationError) as raised:
            OpenCVRecordedSource(
                self.source_path,
                "recorded-file",
                self.policy,
                capture_factory=changing_factory,
                opencv_facade=FakeOpenCV(),
            )

        self.assertEqual(raised.exception.code, SourceFailureCode.SOURCE_CHANGED)
        self.assertEqual(capture.read_count, 0)
        self.assertEqual(capture.release_count, 1)

    def test_source_change_during_decode_fails_final_identity_and_releases_once(self) -> None:
        frame = self.frame()

        def mutate_on_first_read(read_index: int) -> None:
            if read_index == 0:
                self.source_path.write_bytes(b"changed during frame decoding")

        capture = FakeCapture(
            [(True, frame), (False, None)],
            frame_count=1.0,
            timestamps_ms=[0.0],
            on_read=mutate_on_first_read,
        )
        source = self.source(capture)
        iterator = source.frames()

        decoded = next(iterator)
        with self.assertRaises(SourceDecodeError) as raised:
            next(iterator)

        self.assertEqual(decoded.frame_metadata.frame.frame_id, 0)
        self.assertEqual(raised.exception.code, SourceFailureCode.SOURCE_CHANGED)
        self.assertEqual(capture.read_count, 2)
        self.assertEqual(capture.release_count, 1)

    def test_source_change_during_final_eof_read_prevents_successful_exhaustion(self) -> None:
        frame = self.frame()

        def mutate_on_eof_read(read_index: int) -> None:
            if read_index == 1:
                self.source_path.write_bytes(b"changed immediately before valid EOF")

        capture = FakeCapture(
            [(True, frame), (False, None)],
            frame_count=1.0,
            timestamps_ms=[0.0],
            on_read=mutate_on_eof_read,
        )
        source = self.source(capture)
        iterator = source.frames()

        next(iterator)
        with self.assertRaises(SourceDecodeError) as raised:
            next(iterator)

        self.assertEqual(raised.exception.code, SourceFailureCode.SOURCE_CHANGED)
        self.assertEqual(capture.read_count, 2)
        self.assertEqual(capture.release_count, 1)

    def test_missing_file_and_directory_are_rejected_without_opening_capture(self) -> None:
        missing = self.temporary_root / "missing.dat"
        directory = self.temporary_root / "directory"
        directory.mkdir()
        capture_called = False

        def forbidden_factory(_: str) -> FakeCapture:
            nonlocal capture_called
            capture_called = True
            raise AssertionError("capture factory must not be called")

        for path, expected_code in (
            (missing, SourceFailureCode.MISSING),
            (directory, SourceFailureCode.NOT_REGULAR_FILE),
        ):
            with self.subTest(path=path.name):
                with self.assertRaises(SourceValidationError) as raised:
                    OpenCVRecordedSource(
                        path,
                        "recorded-file",
                        self.policy,
                        capture_factory=forbidden_factory,
                        opencv_facade=FakeOpenCV(),
                    )
                self.assertEqual(raised.exception.code, expected_code)
        self.assertFalse(capture_called)

    def test_symlink_is_rejected_when_supported(self) -> None:
        link = self.temporary_root / "source-link.dat"
        try:
            link.symlink_to(self.source_path)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

        with self.assertRaises(SourceValidationError) as raised:
            self.source(FakeCapture([], frame_count=0.0), source_path=link)

        self.assertEqual(raised.exception.code, SourceFailureCode.SYMLINK_FORBIDDEN)

    def test_direct_indirection_rejection_has_deterministic_coverage(self) -> None:
        candidate = self.source_path.absolute()

        with patch.object(
            opencv_recorded,
            "_single_path_is_symlink_or_reparse",
            side_effect=lambda path: path == candidate,
        ):
            with self.assertRaises(SourceValidationError) as raised:
                self.source(FakeCapture([], frame_count=0.0))

        self.assertEqual(raised.exception.code, SourceFailureCode.SYMLINK_FORBIDDEN)

    def test_reparse_parent_rejection_has_deterministic_coverage(self) -> None:
        reparse_component = self.temporary_root.absolute()

        with patch.object(
            opencv_recorded,
            "_single_path_is_symlink_or_reparse",
            side_effect=lambda path: path == reparse_component,
        ):
            with self.assertRaises(SourceValidationError) as raised:
                self.source(FakeCapture([], frame_count=0.0))

        self.assertEqual(raised.exception.code, SourceFailureCode.SYMLINK_FORBIDDEN)

    def test_capture_open_failure_releases_exactly_once(self) -> None:
        capture = FakeCapture([], opened=False)

        with self.assertRaises(SourceValidationError) as raised:
            self.source(capture)

        self.assertEqual(raised.exception.code, SourceFailureCode.CAPTURE_OPEN_FAILED)
        self.assertEqual(capture.release_count, 1)

    def test_valid_reported_metadata(self) -> None:
        capture = FakeCapture([], fps=29.97, frame_count=0.0)
        source = self.source(capture)
        self.addCleanup(source.close)

        self.assertEqual(source.metadata.validated_fps, 29.97)
        self.assertEqual(source.metadata.validated_reported_frame_count, 0)
        self.assertEqual(source.metadata.reported_width, 3)
        self.assertEqual(source.metadata.reported_height, 2)
        self.assertEqual(source.metadata.capture_backend, "FAKE")
        self.assertEqual(source.metadata.codec_fourcc, "MJPG")
        self.assertEqual(source.metadata.unavailable_metadata, ())

    def test_unavailable_backend_and_fourcc_are_explicit(self) -> None:
        capture = FakeCapture([], frame_count=0.0, backend="", codec=0.0)
        source = self.source(capture)
        self.addCleanup(source.close)
        reasons = {item.field: item.reason for item in source.metadata.unavailable_metadata}

        self.assertIsNone(source.metadata.capture_backend)
        self.assertIsNone(source.metadata.codec_fourcc)
        self.assertEqual(
            reasons[UnavailableMetadataField.CAPTURE_BACKEND],
            "not_reported",
        )
        self.assertEqual(reasons[UnavailableMetadataField.CODEC_FOURCC], "not_reported")

    def test_invalid_fps_values_fail_before_iteration(self) -> None:
        for invalid_fps in (0.0, -1.0, float("nan"), float("inf"), "30.0", True):
            with self.subTest(fps=invalid_fps):
                capture = FakeCapture([], fps=invalid_fps, frame_count=0.0)
                with self.assertRaises(SourceValidationError) as raised:
                    self.source(capture)
                self.assertEqual(raised.exception.code, SourceFailureCode.INVALID_FPS)
                self.assertEqual(capture.read_count, 0)
                self.assertEqual(capture.release_count, 1)

    def test_invalid_reported_count_values_fail_before_iteration(self) -> None:
        invalid_counts = (-1.0, float("nan"), float("inf"), 1.5, True, "1")
        for invalid_count in invalid_counts:
            with self.subTest(frame_count=invalid_count):
                capture = FakeCapture([], frame_count=invalid_count)
                with self.assertRaises(SourceValidationError) as raised:
                    self.source(capture)
                self.assertEqual(raised.exception.code, SourceFailureCode.INVALID_FRAME_COUNT)
                self.assertEqual(capture.read_count, 0)
                self.assertEqual(capture.release_count, 1)

    def test_required_metadata_exception_releases_open_capture_once(self) -> None:
        capture = FakeCapture(
            [],
            frame_count=0.0,
            property_errors={FakeOpenCV.CAP_PROP_FPS},
        )

        with self.assertRaises(SourceValidationError) as raised:
            self.source(capture)

        self.assertEqual(raised.exception.code, SourceFailureCode.CAPTURE_OPEN_FAILED)
        self.assertEqual(capture.release_count, 1)

    def test_optional_backend_exception_becomes_unavailable(self) -> None:
        capture = FakeCapture(
            [],
            frame_count=0.0,
            backend_error=OSError("controlled optional backend failure"),
        )
        source = self.source(capture)
        reasons = {item.field: item.reason for item in source.metadata.unavailable_metadata}

        self.assertIsNone(source.metadata.capture_backend)
        self.assertEqual(reasons[UnavailableMetadataField.CAPTURE_BACKEND], "unsupported")
        source.close()
        self.assertEqual(capture.release_count, 1)

    def test_optional_fourcc_exception_becomes_unavailable(self) -> None:
        capture = FakeCapture(
            [],
            frame_count=0.0,
            property_errors={FakeOpenCV.CAP_PROP_FOURCC},
        )
        source = self.source(capture)
        reasons = {item.field: item.reason for item in source.metadata.unavailable_metadata}

        self.assertIsNone(source.metadata.codec_fourcc)
        self.assertEqual(reasons[UnavailableMetadataField.CODEC_FOURCC], "unsupported")
        source.close()
        self.assertEqual(capture.release_count, 1)

    def test_domain_metadata_construction_failure_releases_open_capture_once(self) -> None:
        capture = FakeCapture([], frame_count=0.0)

        with patch.object(
            opencv_recorded,
            "RecordedSourceMetadata",
            side_effect=ValueError("controlled domain metadata failure"),
        ):
            with self.assertRaisesRegex(ValueError, "controlled domain metadata failure"):
                self.source(capture)

        self.assertEqual(capture.release_count, 1)

    def test_unavailable_container_dimensions_do_not_override_decoded_dimensions(self) -> None:
        frame = self.frame(height=4, width=5)
        capture = FakeCapture(
            [(True, frame), (False, None)],
            frame_count=1.0,
            width=0.0,
            height=float("nan"),
            timestamps_ms=[0.0],
        )
        source = self.source(capture)

        decoded = list(source.frames())[0]
        reasons = {item.field: item.reason for item in source.metadata.unavailable_metadata}

        self.assertIsNone(source.metadata.reported_width)
        self.assertIsNone(source.metadata.reported_height)
        self.assertEqual(reasons[UnavailableMetadataField.REPORTED_WIDTH], "not_reported")
        self.assertEqual(reasons[UnavailableMetadataField.REPORTED_HEIGHT], "invalid")
        self.assertEqual(decoded.frame_metadata.decoded_width, 5)
        self.assertEqual(decoded.frame_metadata.decoded_height, 4)

    def test_frame_ids_canonical_timestamps_and_exact_eof_reads(self) -> None:
        frames = [self.frame(), self.frame() + 1]
        capture = FakeCapture(
            [(True, frame) for frame in frames] + [(False, None)],
            fps=2.0,
            frame_count=2.0,
            timestamps_ms=[100.0, 600.0],
        )
        source = self.source(capture)

        decoded = list(source.frames())

        self.assertEqual(source.metadata.source.source_id, "recorded-file")
        self.assertNotEqual(source.metadata.source.source_id, self.source_path.stem)
        self.assertEqual([item.frame_metadata.frame.frame_id for item in decoded], [0, 1])
        self.assertTrue(
            all(item.frame_metadata.frame.source_id == "recorded-file" for item in decoded)
        )
        self.assertEqual(
            [item.frame_metadata.frame.timestamp_s for item in decoded],
            [0.0, 0.5],
        )
        self.assertEqual(capture.read_count, 3)
        self.assertEqual(capture.release_count, 1)

    def test_diagnostic_timestamp_recovery_uses_last_increasing_baseline(self) -> None:
        frames = [self.frame() + index for index in range(6)]
        capture = FakeCapture(
            [(True, frame) for frame in frames] + [(False, None)],
            frame_count=6.0,
            timestamps_ms=[
                100.0,
                90.0,
                float("nan"),
                95.0,
                100.0,
                110.0,
            ],
        )
        source = self.source(capture)

        decoded = list(source.frames())

        self.assertEqual(
            [item.frame_metadata.source_timestamp_status for item in decoded],
            [
                SourceTimestampStatus.AVAILABLE,
                SourceTimestampStatus.REGRESSIVE,
                SourceTimestampStatus.UNAVAILABLE,
                SourceTimestampStatus.REGRESSIVE,
                SourceTimestampStatus.DUPLICATED,
                SourceTimestampStatus.AVAILABLE,
            ],
        )
        self.assertEqual(
            [item.frame_metadata.source_timestamp_ms for item in decoded],
            [100.0, 90.0, None, 95.0, 100.0, 110.0],
        )
        self.assertEqual(
            [item.frame_metadata.frame.timestamp_s for item in decoded],
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        )

    def test_negative_and_nonfinite_diagnostic_timestamps_are_unavailable(self) -> None:
        for diagnostic_value in (-1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(diagnostic_value=diagnostic_value):
                capture = FakeCapture(
                    [(True, self.frame()), (False, None)],
                    frame_count=1.0,
                    timestamps_ms=[diagnostic_value],
                )
                source = self.source(capture)

                decoded = list(source.frames())[0]

                self.assertEqual(
                    decoded.frame_metadata.source_timestamp_status,
                    SourceTimestampStatus.UNAVAILABLE,
                )
                self.assertIsNone(decoded.frame_metadata.source_timestamp_ms)

    def test_early_decode_failure_is_not_eof_and_closes_capture(self) -> None:
        capture = FakeCapture(
            [(True, self.frame()), (False, None)],
            frame_count=2.0,
            timestamps_ms=[0.0],
        )
        source = self.source(capture)

        with self.assertRaises(SourceDecodeError) as raised:
            list(source.frames())

        self.assertEqual(raised.exception.code, SourceFailureCode.EARLY_DECODE_FAILURE)
        self.assertEqual(capture.read_count, 2)
        self.assertEqual(capture.release_count, 1)

    def test_extra_frame_is_count_mismatch_and_is_not_yielded(self) -> None:
        first = self.frame()
        extra = self.frame() + 1
        capture = FakeCapture(
            [(True, first), (True, extra)],
            frame_count=1.0,
            timestamps_ms=[0.0],
        )
        source = self.source(capture)
        iterator = source.frames()

        decoded = next(iterator)
        with self.assertRaises(SourceDecodeError) as raised:
            next(iterator)

        self.assertIsNot(decoded.bgr, first)
        self.assertTrue(first.flags.writeable)
        self.assertEqual(raised.exception.code, SourceFailureCode.FRAME_COUNT_MISMATCH)
        self.assertEqual(capture.read_count, 2)
        self.assertEqual(capture.release_count, 1)

    def test_iteration_is_one_shot_and_close_is_idempotent(self) -> None:
        capture = FakeCapture([(False, None)], frame_count=0.0)
        source = self.source(capture)

        self.assertEqual(list(source.frames()), [])
        with self.assertRaises(SourceLifecycleError) as raised:
            source.frames()
        source.close()
        source.close()

        self.assertEqual(raised.exception.code, SourceFailureCode.ALREADY_CONSUMED)
        self.assertEqual(capture.read_count, 1)
        self.assertEqual(capture.release_count, 1)

    def test_unstarted_iterator_close_releases_capture_once(self) -> None:
        capture = FakeCapture([(False, None)], frame_count=0.0)
        source = self.source(capture)
        iterator = source.frames()

        iterator.close()
        iterator.close()

        self.assertEqual(capture.read_count, 0)
        self.assertEqual(capture.release_count, 1)
        with self.assertRaises(SourceLifecycleError):
            next(iterator)

    def test_two_iterator_requests_before_first_next_cannot_both_succeed(self) -> None:
        capture = FakeCapture([(False, None)], frame_count=0.0)
        source = self.source(capture)
        first = source.frames()

        with self.assertRaises(SourceLifecycleError) as raised:
            source.frames()

        self.assertEqual(raised.exception.code, SourceFailureCode.ALREADY_CONSUMED)
        first.close()
        self.assertEqual(capture.release_count, 1)

    def test_partial_iteration_then_iterator_close_releases_once(self) -> None:
        capture = FakeCapture(
            [(True, self.frame()), (True, self.frame() + 1), (False, None)],
            frame_count=2.0,
            timestamps_ms=[0.0, 1.0],
        )
        source = self.source(capture)
        iterator = source.frames()

        first = next(iterator)
        iterator.close()
        iterator.close()

        self.assertEqual(first.frame_metadata.frame.frame_id, 0)
        self.assertEqual(capture.read_count, 1)
        self.assertEqual(capture.release_count, 1)
        with self.assertRaises(SourceLifecycleError):
            next(iterator)

    def test_frames_after_manual_close_fail_explicitly(self) -> None:
        source = self.source(FakeCapture([], frame_count=0.0))
        source.close()

        with self.assertRaises(SourceLifecycleError) as raised:
            source.frames()

        self.assertEqual(raised.exception.code, SourceFailureCode.CLOSED)

    def test_context_manager_closes_without_iteration(self) -> None:
        capture = FakeCapture([], frame_count=0.0)
        source = self.source(capture)

        with source as entered:
            self.assertIs(entered, source)

        self.assertEqual(capture.release_count, 1)

    def test_source_context_exit_closes_unstarted_iterator(self) -> None:
        capture = FakeCapture([(False, None)], frame_count=0.0)
        source = self.source(capture)

        with source:
            iterator = source.frames()

        self.assertEqual(capture.read_count, 0)
        self.assertEqual(capture.release_count, 1)
        with self.assertRaises(SourceLifecycleError):
            next(iterator)

    def test_invalid_frame_dtype_and_shape_close_capture(self) -> None:
        invalid_frames = (
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros((2, 3, 4), dtype=np.uint8),
        )
        for invalid_frame in invalid_frames:
            with self.subTest(shape=invalid_frame.shape, dtype=invalid_frame.dtype):
                capture = FakeCapture(
                    [(True, invalid_frame)],
                    frame_count=1.0,
                    timestamps_ms=[0.0],
                )
                source = self.source(capture)
                with self.assertRaises(SourceDecodeError) as raised:
                    list(source.frames())
                self.assertEqual(raised.exception.code, SourceFailureCode.INVALID_FRAME)
                self.assertEqual(capture.release_count, 1)

    def test_non_contiguous_frame_is_normalized_and_hash_is_correct(self) -> None:
        base = self.frame(width=4)
        non_contiguous = base[:, ::-1, :]
        expected_pixels = non_contiguous.copy()
        self.assertFalse(non_contiguous.flags.c_contiguous)
        capture = FakeCapture(
            [(True, non_contiguous), (False, None)],
            frame_count=1.0,
            width=4.0,
            timestamps_ms=[0.0],
        )
        source = self.source(capture)

        decoded = list(source.frames())[0]
        expected_hash = hashlib.sha256(decoded.bgr.tobytes(order="C")).hexdigest()

        self.assertTrue(decoded.bgr.flags.c_contiguous)
        self.assertIsNot(decoded.bgr, non_contiguous)
        self.assertTrue(decoded.bgr.flags.owndata)
        self.assertIsNone(decoded.bgr.base)
        self.assertEqual(decoded.frame_metadata.decoded_bgr_sha256, expected_hash)
        self.assertFalse(decoded.bgr.flags.writeable)
        self.assertTrue(non_contiguous.flags.writeable)
        base.fill(255)
        np.testing.assert_array_equal(decoded.bgr, expected_pixels)

    def test_contiguous_owned_capture_frame_is_copied_to_owned_immutable_storage(self) -> None:
        frame = self.frame().copy()
        expected_pixels = frame.copy()
        self.assertTrue(frame.flags.owndata)
        capture = FakeCapture(
            [(True, frame), (False, None)],
            frame_count=1.0,
            timestamps_ms=[0.0],
        )
        source = self.source(capture)

        decoded = list(source.frames())[0]

        self.assertIsNot(decoded.bgr, frame)
        self.assertTrue(decoded.bgr.flags.c_contiguous)
        self.assertTrue(decoded.bgr.flags.owndata)
        self.assertIsNone(decoded.bgr.base)
        self.assertFalse(decoded.bgr.flags.writeable)
        self.assertTrue(frame.flags.writeable)
        frame.fill(255)
        np.testing.assert_array_equal(decoded.bgr, expected_pixels)
        self.assertEqual(
            decoded.frame_metadata.decoded_bgr_sha256,
            hashlib.sha256(memoryview(decoded.bgr).cast("B")).hexdigest(),
        )

    def test_contiguous_view_is_detached_from_mutable_base(self) -> None:
        base = np.arange(18, dtype=np.uint8)
        contiguous_view = base.reshape(2, 3, 3)
        expected_pixels = contiguous_view.copy()
        self.assertTrue(contiguous_view.flags.c_contiguous)
        self.assertFalse(contiguous_view.flags.owndata)
        capture = FakeCapture(
            [(True, contiguous_view), (False, None)],
            frame_count=1.0,
            timestamps_ms=[0.0],
        )
        source = self.source(capture)

        decoded = list(source.frames())[0]
        base.fill(255)

        np.testing.assert_array_equal(decoded.bgr, expected_pixels)
        self.assertTrue(decoded.bgr.flags.owndata)
        self.assertIsNone(decoded.bgr.base)
        self.assertFalse(decoded.bgr.flags.writeable)
        self.assertTrue(contiguous_view.flags.writeable)
        self.assertEqual(
            decoded.frame_metadata.decoded_bgr_sha256,
            hashlib.sha256(memoryview(decoded.bgr).cast("B")).hexdigest(),
        )

    def test_capture_can_reuse_same_mutable_array_without_frame_aliasing(self) -> None:
        first_payload = np.full((2, 3, 3), 10, dtype=np.uint8)
        second_payload = np.full((2, 3, 3), 20, dtype=np.uint8)
        capture = ReusingBufferCapture([first_payload, second_payload])
        source = self.source(capture)

        decoded = list(source.frames())

        np.testing.assert_array_equal(decoded[0].bgr, first_payload)
        np.testing.assert_array_equal(decoded[1].bgr, second_payload)
        self.assertIsNot(decoded[0].bgr, decoded[1].bgr)
        self.assertTrue(capture.shared_buffer.flags.writeable)
        self.assertTrue(all(frame.bgr.flags.owndata for frame in decoded))
        self.assertTrue(all(frame.bgr.base is None for frame in decoded))
        self.assertTrue(all(not frame.bgr.flags.writeable for frame in decoded))

    def test_production_frame_construction_hashes_owned_bytes_once(self) -> None:
        frame = self.frame()
        capture = FakeCapture(
            [(True, frame), (False, None)],
            frame_count=1.0,
            timestamps_ms=[0.0],
        )
        source = self.source(capture)

        with patch.object(
            source_ports,
            "_sha256_bgr",
            wraps=source_ports._sha256_bgr,
        ) as hash_bgr:
            decoded = list(source.frames())[0]

        self.assertEqual(hash_bgr.call_count, 1)
        self.assertEqual(
            decoded.frame_metadata.decoded_bgr_sha256,
            hashlib.sha256(memoryview(decoded.bgr).cast("B")).hexdigest(),
        )

    def test_decoded_frame_rejects_caller_supplied_metadata_construction(self) -> None:
        frame = self.frame()
        digest = hashlib.sha256(memoryview(frame).cast("B")).hexdigest()
        frame_ref = FrameRef(source_id="recorded-file", frame_id=0, timestamp_s=0.0)
        wrong_dimensions = FrameMetadata(
            frame=frame_ref,
            decoded_width=4,
            decoded_height=2,
            source_timestamp_ms=None,
            source_timestamp_status=SourceTimestampStatus.UNAVAILABLE,
            decoded_bgr_sha256=digest,
        )
        wrong_hash = FrameMetadata(
            frame=frame_ref,
            decoded_width=3,
            decoded_height=2,
            source_timestamp_ms=None,
            source_timestamp_status=SourceTimestampStatus.UNAVAILABLE,
            decoded_bgr_sha256="0" * 64,
        )

        with self.assertRaises(TypeError):
            DecodedFrame()
        with self.assertRaises(TypeError):
            DecodedFrame(frame_metadata=wrong_dimensions, bgr=frame.copy())
        with self.assertRaises(TypeError):
            DecodedFrame(frame_metadata=wrong_hash, bgr=frame.copy())

    def test_artifact_contracts_reject_noncanonical_and_coercive_values(self) -> None:
        valid = {
            "source_id": "recorded-file",
            "sha256": "a" * 64,
            "byte_size": 1,
        }
        for field_name, invalid_value in (
            ("sha256", "A" * 64),
            ("byte_size", True),
            ("byte_size", "1"),
            ("byte_size", -1),
        ):
            with self.subTest(field=field_name, value=invalid_value):
                data = dict(valid)
                data[field_name] = invalid_value
                with self.assertRaises(ValidationError):
                    SourceFingerprint.model_validate(data)
        with self.assertRaises(ValidationError):
            SourceFingerprint.model_validate({**valid, "unknown": 1})

        metadata_source = self.source(FakeCapture([], frame_count=0.0))
        self.addCleanup(metadata_source.close)
        metadata_data = metadata_source.metadata.model_dump(mode="python")
        for field_name, invalid_value in (
            ("reported_width", True),
            ("reported_height", "2"),
            ("validated_fps", True),
            ("validated_fps", "2.0"),
            ("validated_fps", float("nan")),
            ("validated_reported_frame_count", True),
            ("validated_reported_frame_count", "0"),
        ):
            with self.subTest(metadata_field=field_name, value=invalid_value):
                invalid_metadata = dict(metadata_data)
                invalid_metadata[field_name] = invalid_value
                with self.assertRaises(ValidationError):
                    type(metadata_source.metadata).model_validate(invalid_metadata)

        for invalid_fourcc in ("A", "AB", "ABC", "ABCDE", "A\nCD", "éBCD"):
            with self.subTest(codec_fourcc=invalid_fourcc):
                invalid_metadata = dict(metadata_data)
                invalid_metadata["codec_fourcc"] = invalid_fourcc
                with self.assertRaises(ValidationError):
                    type(metadata_source.metadata).model_validate(invalid_metadata)

        frame_data = {
            "frame": {"source_id": "recorded-file", "frame_id": 0, "timestamp_s": 0.0},
            "decoded_width": 3,
            "decoded_height": 2,
            "source_timestamp_ms": None,
            "source_timestamp_status": SourceTimestampStatus.UNAVAILABLE,
            "decoded_bgr_sha256": "a" * 64,
        }
        for field_name, invalid_value in (
            ("decoded_width", True),
            ("decoded_height", "2"),
            ("source_timestamp_ms", float("inf")),
        ):
            with self.subTest(frame_metadata_field=field_name, value=invalid_value):
                invalid_frame_metadata = dict(frame_data)
                invalid_frame_metadata[field_name] = invalid_value
                with self.assertRaises(ValidationError):
                    FrameMetadata.model_validate(invalid_frame_metadata)
        for nested_field, invalid_value in (("frame_id", True), ("timestamp_s", "0.0")):
            with self.subTest(frame_ref_field=nested_field, value=invalid_value):
                invalid_frame_metadata = dict(frame_data)
                invalid_frame_metadata["frame"] = dict(frame_data["frame"])
                invalid_frame_metadata["frame"][nested_field] = invalid_value
                with self.assertRaises(ValidationError):
                    FrameMetadata.model_validate(invalid_frame_metadata)

    def test_real_video_capture_is_never_invoked_when_factory_is_injected(self) -> None:
        capture = FakeCapture([], frame_count=0.0)
        with patch.object(
            opencv_recorded.cv2,
            "VideoCapture",
            side_effect=AssertionError("real capture must not be invoked"),
        ):
            source = self.source(capture)
            source.close()

        self.assertEqual(capture.release_count, 1)

    def test_source_errors_do_not_include_absolute_path(self) -> None:
        missing = self.temporary_root / "private-user-path" / "missing.dat"

        with self.assertRaises(SourceValidationError) as raised:
            self.source(FakeCapture([]), source_path=missing)

        self.assertNotIn(str(self.temporary_root), str(raised.exception))
        self.assertNotIn(str(self.temporary_root), raised.exception.code.value)
        self.assertNotIn(str(self.temporary_root), raised.exception.detail)
        serialized_public_error = json.dumps(
            {
                "code": raised.exception.code.value,
                "detail": raised.exception.detail,
            }
        )
        self.assertNotIn(str(self.temporary_root), serialized_public_error)

    def test_chained_operational_path_is_excluded_from_public_error_fields(self) -> None:
        capture = FakeCapture([], frame_count=0.0)
        private_message = f"metadata failed for {self.source_path}"

        with patch.object(capture, "get", side_effect=OSError(private_message)):
            with self.assertRaises(SourceValidationError) as raised:
                self.source(capture)

        self.assertIsNotNone(raised.exception.__cause__)
        self.assertIn(str(self.source_path), str(raised.exception.__cause__))
        serialized_public_error = json.dumps(
            {
                "code": raised.exception.code.value,
                "detail": raised.exception.detail,
            }
        )
        self.assertNotIn(str(self.temporary_root), serialized_public_error)
        self.assertEqual(capture.release_count, 1)


if __name__ == "__main__":
    unittest.main()
