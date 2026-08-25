from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from geovision.core.config import Settings
from geovision.core.experiment_config import load_m1_experiment_config
from geovision.domain.benchmark import (
    M1ExperimentConfig,
    ReadinessStatus,
    SourceType,
    TrackerProfile,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "m1-benchmark.yaml"


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_m1_experiment_config(CONFIG_PATH)

    def config_data(self) -> dict[str, object]:
        return copy.deepcopy(self.config.model_dump(mode="json"))

    def test_config_resolves_exact_four_way_matrix(self) -> None:
        combinations = self.config.resolve_combinations()

        self.assertEqual(
            [combination.combination_id for combination in combinations],
            [
                "yolo11n-bytetrack",
                "yolo11n-botsort-sparse-optical-flow",
                "yolo26n-bytetrack",
                "yolo26n-botsort-sparse-optical-flow",
            ],
        )
        self.assertEqual(len({item.combination_id for item in combinations}), 4)

    def test_sources_preserve_file_matrix_and_live_smoke_types(self) -> None:
        source_by_type = {source.source_type: source for source in self.config.sources}

        self.assertTrue(source_by_type[SourceType.FILE].comparative)
        self.assertFalse(source_by_type[SourceType.WEBCAM].comparative)
        self.assertFalse(source_by_type[SourceType.RTSP].comparative)

    def test_detector_comparison_semantics_are_exact_and_shared(self) -> None:
        detector_by_id = {detector.profile_id: detector for detector in self.config.detectors}

        self.assertEqual(
            self.config.detector_comparison.model_dump(mode="json"),
            {
                "task": "detect",
                "image_size": 640,
                "confidence": 0.10,
                "iou": 0.70,
                "max_det": 300,
                "rect": True,
                "classes": None,
                "agnostic_nms": False,
                "augment": False,
                "end_to_end": "model-native",
                "compile": False,
                "channels_last": False,
            },
        )
        self.assertEqual(detector_by_id["yolo11n"].model, "yolo11n.pt")
        self.assertEqual(detector_by_id["yolo11n"].weights_size_bytes, 5_613_764)
        self.assertEqual(detector_by_id["yolo26n"].model, "yolo26n.pt")
        self.assertEqual(detector_by_id["yolo26n"].weights_size_bytes, 5_544_453)
        for detector in self.config.detectors:
            self.assertNotIn("image_size", type(detector).model_fields)
            self.assertNotIn("confidence", type(detector).model_fields)

    def test_tracker_profiles_lock_exact_arguments_gmc_and_reid(self) -> None:
        tracker_by_backend = {tracker.backend: tracker for tracker in self.config.trackers}
        expected_common = {
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.25,
            "track_buffer": 30,
            "track_buffer_unit": "frames",
            "match_thresh": 0.80,
            "fuse_score": True,
            "reid": False,
        }

        for tracker in tracker_by_backend.values():
            for field_name, expected in expected_common.items():
                self.assertEqual(getattr(tracker, field_name), expected)

        bytetrack = tracker_by_backend["bytetrack"]
        self.assertIsNone(bytetrack.gmc_method)
        self.assertIsNone(bytetrack.proximity_thresh)
        self.assertIsNone(bytetrack.appearance_thresh)
        self.assertIsNone(bytetrack.model)

        botsort = tracker_by_backend["botsort"]
        self.assertEqual(botsort.gmc_method, "sparseOptFlow")
        self.assertEqual(botsort.proximity_thresh, 0.50)
        self.assertEqual(botsort.appearance_thresh, 0.80)
        self.assertEqual(botsort.model, "auto")

    def test_runtime_and_automatic_install_policy_are_locked(self) -> None:
        self.assertEqual(self.config.runtime.device, 0)
        self.assertEqual(self.config.runtime.batch_size, 1)
        self.assertTrue(self.config.runtime.fp16)
        self.assertFalse(self.config.runtime.automatic_dependency_install)
        self.assertFalse(self.config.runtime.automatic_model_download)
        self.assertEqual(self.config.measurement.warmup_frames, 30)

    def test_recorded_source_policy_is_exact(self) -> None:
        policy = self.config.recorded_source

        self.assertEqual(policy.frame_index_origin, 0)
        self.assertTrue(policy.contiguous_frame_indices)
        self.assertEqual(policy.canonical_timestamp, "frame-index-over-validated-fps")
        self.assertEqual(policy.source_timestamp_role, "diagnostic")
        self.assertEqual(
            policy.source_timestamp_statuses,
            ("available", "unavailable", "duplicated", "regressive"),
        )
        self.assertTrue(policy.require_finite_positive_fps)
        self.assertTrue(policy.invalid_fps_fails_before_inference)
        self.assertTrue(policy.sequential_decode)
        self.assertFalse(policy.allow_seek)
        self.assertFalse(policy.allow_skip)
        self.assertFalse(policy.allow_retry)
        self.assertTrue(policy.require_reported_frame_count_match)
        self.assertEqual(policy.early_decode_failure, "fail-run")
        self.assertEqual(policy.ambiguous_eof_without_frame_count, "fail-run")
        self.assertEqual((policy.pixel_format, policy.pixel_dtype), ("BGR", "uint8"))
        self.assertEqual((policy.channel_layout, policy.channel_count), ("HWC", 3))
        self.assertEqual(policy.dimension_authority, "decoded-frame")
        self.assertEqual(policy.source_fingerprint, "sha256-and-byte-size")
        self.assertEqual(policy.replay_frame_fingerprint, "sha256")
        self.assertFalse(policy.source_paths_in_artifacts)

    def test_artifact_identifiers_and_canonical_policy_are_exact(self) -> None:
        self.assertEqual(
            self.config.artifacts.model_dump(mode="json"),
            {
                "frame_metadata": "geovision.frame-metadata/v1",
                "detection_cache": "geovision.detection-cache/v1",
                "track_artifact": "geovision.track-artifact/v1",
                "runtime_telemetry": "geovision.runtime-telemetry/v1",
                "experiment_manifest": "geovision.experiment-manifest/v1",
                "failure_report": "geovision.failure-report/v1",
                "completion_marker": "geovision.completion-marker/v1",
                "mot_sidecar": "geovision.mot-sidecar/v1",
                "detection_cache_format": "jsonl",
                "manifest_format": "json",
            },
        )
        self.assertEqual(
            self.config.canonical_artifacts.model_dump(mode="json"),
            {
                "detection_order": [
                    "confidence-desc",
                    "class-id-asc",
                    "x1-asc",
                    "y1-asc",
                    "x2-asc",
                    "y2-asc",
                    "original-ordinal-asc",
                ],
                "external_scalar_conversion": "python-numeric",
                "float_representation": "python-3.11-shortest-round-trip",
                "preserve_full_finite_precision": True,
                "normalize_negative_zero": True,
                "encoding": "utf-8",
                "sort_keys": True,
                "compact_separators": True,
                "allow_nan": False,
            },
        )

    def test_unknown_fields_are_rejected(self) -> None:
        data = self.config_data()
        data["unexpected"] = True

        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(data)

        tracker_data = self.config_data()
        tracker_data["trackers"][0]["unexpected"] = True  # type: ignore[index]
        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(tracker_data)

    def test_invalid_device_batch_image_precision_and_warmup_are_rejected(self) -> None:
        invalid_values = (
            ("runtime", "device", "cuda:0"),
            ("runtime", "device", False),
            ("runtime", "batch_size", 2),
            ("runtime", "batch_size", True),
            ("runtime", "fp16", False),
            ("runtime", "fp16", 1),
            ("runtime", "automatic_dependency_install", True),
            ("runtime", "automatic_model_download", True),
            ("detector_comparison", "image_size", 320),
            ("detector_comparison", "image_size", 640.0),
            ("detector_comparison", "confidence", "0.10"),
            ("measurement", "warmup_frames", 29),
            ("recorded_source", "require_reported_frame_count_match", False),
        )

        for section, field_name, invalid_value in invalid_values:
            with self.subTest(field=field_name):
                data = self.config_data()
                data[section][field_name] = invalid_value  # type: ignore[index]
                with self.assertRaises(ValidationError):
                    M1ExperimentConfig.model_validate(data)

    def test_detector_confidence_above_tracker_low_threshold_is_rejected(self) -> None:
        data = self.config_data()
        data["detector_comparison"]["confidence"] = 0.11  # type: ignore[index]

        with self.assertRaisesRegex(
            ValidationError,
            "detector confidence cannot exceed tracker low threshold",
        ):
            M1ExperimentConfig.model_validate(data)

    def test_detector_model_pairing_is_exact(self) -> None:
        data = self.config_data()
        data["detectors"][0]["model"] = "yolo26n.pt"  # type: ignore[index]

        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(data)

    def test_checkpoint_hashes_must_be_canonical_lowercase_sha256(self) -> None:
        invalid_hashes = (
            "0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1",
            "a" * 63,
        )

        for invalid_hash in invalid_hashes:
            with self.subTest(hash=invalid_hash):
                data = self.config_data()
                data["detectors"][0]["weights_sha256"] = invalid_hash  # type: ignore[index]
                with self.assertRaises(ValidationError):
                    M1ExperimentConfig.model_validate(data)

    def test_tracker_threshold_order_ranges_and_buffer_are_rejected(self) -> None:
        invalid_mutations = (
            ("track_low_thresh", 0.26),
            ("track_high_thresh", 1.01),
            ("new_track_thresh", float("nan")),
            ("track_buffer", 0),
            ("track_buffer_unit", "seconds"),
            ("match_thresh", "0.80"),
            ("match_thresh", float("inf")),
        )

        for field_name, invalid_value in invalid_mutations:
            with self.subTest(field=field_name):
                data = self.config_data()
                data["trackers"][0][field_name] = invalid_value  # type: ignore[index]
                with self.assertRaises(ValidationError):
                    M1ExperimentConfig.model_validate(data)

    def test_botsort_only_thresholds_reject_invalid_scalar_inputs(self) -> None:
        invalid_values = (
            True,
            "0.50",
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.01,
            1.01,
        )

        for field_name in ("proximity_thresh", "appearance_thresh"):
            for invalid_value in invalid_values:
                with self.subTest(
                    validation="tracker-profile",
                    field=field_name,
                    value=invalid_value,
                ):
                    profile_data = self.config.trackers[1].model_dump(mode="json")
                    profile_data[field_name] = invalid_value
                    with self.assertRaises(ValidationError):
                        TrackerProfile.model_validate(profile_data)

                with self.subTest(
                    validation="m1-config",
                    field=field_name,
                    value=invalid_value,
                ):
                    config_data = self.config_data()
                    config_data["trackers"][1][field_name] = invalid_value  # type: ignore[index]
                    with self.assertRaises(ValidationError):
                        M1ExperimentConfig.model_validate(config_data)

    def test_standalone_botsort_thresholds_accept_inclusive_boundaries(self) -> None:
        for field_name in ("proximity_thresh", "appearance_thresh"):
            for boundary in (0.0, 1.0):
                with self.subTest(field=field_name, boundary=boundary):
                    profile_data = self.config.trackers[1].model_dump(mode="json")
                    profile_data[field_name] = boundary

                    profile = TrackerProfile.model_validate(profile_data)

                    self.assertEqual(getattr(profile, field_name), boundary)

    def test_bytetrack_forbids_gmc_reid_and_botsort_only_fields(self) -> None:
        invalid_mutations = (
            ("gmc_method", "sparseOptFlow"),
            ("reid", True),
            ("proximity_thresh", 0.50),
            ("appearance_thresh", 0.80),
            ("model", "auto"),
        )

        for field_name, invalid_value in invalid_mutations:
            with self.subTest(field=field_name):
                data = self.config_data()
                data["trackers"][0][field_name] = invalid_value  # type: ignore[index]
                with self.assertRaises(ValidationError):
                    M1ExperimentConfig.model_validate(data)

    def test_botsort_requires_sparse_optical_flow_and_forbids_reid(self) -> None:
        invalid_mutations = (
            ("gmc_method", None),
            ("reid", True),
            ("proximity_thresh", None),
            ("appearance_thresh", None),
            ("model", None),
        )

        for field_name, invalid_value in invalid_mutations:
            with self.subTest(field=field_name):
                data = self.config_data()
                data["trackers"][1][field_name] = invalid_value  # type: ignore[index]
                with self.assertRaises(ValidationError):
                    M1ExperimentConfig.model_validate(data)

    def test_enabled_deferred_service_is_rejected(self) -> None:
        data = self.config_data()
        data["services"]["mission_memory"] = True  # type: ignore[index]

        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(data)

    def test_live_source_cannot_join_comparative_matrix(self) -> None:
        data = self.config_data()
        data["sources"][1]["comparative"] = True  # type: ignore[index]

        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(data)

    def test_verified_config_is_execution_ready(self) -> None:
        readiness = self.config.execution_readiness()
        detector_by_id = {detector.profile_id: detector for detector in self.config.detectors}

        self.assertEqual(self.config.runtime.ultralytics_version, "8.4.127")
        self.assertEqual(self.config.runtime.torch_version, "2.13.0+cu130")
        self.assertEqual(self.config.runtime.cuda_version, "13.0")
        self.assertTrue(self.config.runtime.hardware_confirmed)
        self.assertEqual(
            detector_by_id["yolo11n"].weights_sha256,
            "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        )
        self.assertEqual(
            detector_by_id["yolo26n"].weights_sha256,
            "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        )
        self.assertEqual(readiness.status, ReadinessStatus.READY)
        self.assertEqual(readiness.unresolved_fields, ())
        self.assertTrue(readiness.ready)
        self.config.require_execution_ready()

    def test_deliberately_unresolved_copy_fails_closed(self) -> None:
        data = self.config_data()
        data["runtime"]["hardware_confirmed"] = False  # type: ignore[index]
        data["detectors"][1]["weights_sha256"] = None  # type: ignore[index]
        unresolved_config = M1ExperimentConfig.model_validate(data)
        readiness = unresolved_config.execution_readiness()

        self.assertEqual(readiness.status, ReadinessStatus.BLOCKED)
        self.assertEqual(
            readiness.unresolved_fields,
            (
                "runtime.hardware_confirmed",
                "detectors.yolo26n.weights_sha256",
            ),
        )
        self.assertFalse(readiness.ready)
        with self.assertRaises(ValueError):
            unresolved_config.require_execution_ready()

    def test_environment_and_application_settings_do_not_change_semantics(self) -> None:
        expected = self.config.canonical_json()
        application_settings = Settings(
            detector_model="not-a-benchmark-model.pt",
            tracker_backend="not-a-benchmark-tracker",
        )

        with patch.dict(
            os.environ,
            {
                "GEOVISION_DETECTOR_MODEL": "environment-model.pt",
                "GEOVISION_TRACKER_BACKEND": "environment-tracker",
            },
        ):
            reloaded = load_m1_experiment_config(CONFIG_PATH)

        self.assertEqual(reloaded.canonical_json(), expected)
        self.assertNotIn(application_settings.detector_model, expected)
        self.assertNotIn(application_settings.tracker_backend, expected)

    def test_serialization_and_publishability_policy_are_deterministic(self) -> None:
        first = self.config.canonical_json()
        second = load_m1_experiment_config(CONFIG_PATH).canonical_json()

        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(json.loads(first), separators=(",", ":"), sort_keys=True),
            first,
        )
        canonical_data = json.loads(first)
        canonical_bytetrack = next(
            tracker
            for tracker in canonical_data["trackers"]
            if tracker["backend"] == "bytetrack"
        )
        self.assertNotIn("gmc_method", canonical_bytetrack)
        self.assertNotIn("proximity_thresh", canonical_bytetrack)
        self.assertNotIn("appearance_thresh", canonical_bytetrack)
        self.assertNotIn("model", canonical_bytetrack)
        self.assertEqual(self.config.measurement.warmup_frames, 30)
        self.assertTrue(
            self.config.measurement.fresh_detector_pass_required_for_publishable_run
        )
        self.assertTrue(self.config.measurement.same_run_resume_cache_allowed)
        self.assertTrue(self.config.measurement.cross_run_cache_allowed_for_development)
        self.assertFalse(self.config.measurement.cross_run_cache_publishable)

    def test_canonical_json_normalizes_negative_zero(self) -> None:
        comparison = self.config.detector_comparison.model_copy(
            update={"confidence": -0.0}
        )
        config = self.config.model_copy(update={"detector_comparison": comparison})

        canonical = config.canonical_json()

        self.assertEqual(json.loads(canonical)["detector_comparison"]["confidence"], 0.0)
        self.assertNotIn("-0.0", canonical)


if __name__ == "__main__":
    unittest.main()
