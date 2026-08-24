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
from geovision.domain.benchmark import M1ExperimentConfig, ReadinessStatus, SourceType

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "m1-benchmark.yaml"


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_m1_experiment_config(CONFIG_PATH)

    def config_data(self) -> dict[str, object]:
        return copy.deepcopy(self.config.model_dump(mode="json"))

    def test_draft_config_resolves_exact_four_way_matrix(self) -> None:
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

    def test_tracker_profiles_lock_gmc_and_disable_reid(self) -> None:
        tracker_by_backend = {tracker.backend: tracker for tracker in self.config.trackers}

        self.assertEqual(tracker_by_backend["bytetrack"].gmc_method, "none")
        self.assertEqual(tracker_by_backend["botsort"].gmc_method, "sparseOptFlow")
        self.assertFalse(tracker_by_backend["bytetrack"].reid)
        self.assertFalse(tracker_by_backend["botsort"].reid)

    def test_unknown_fields_are_rejected(self) -> None:
        data = self.config_data()
        data["unexpected"] = True

        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(data)

    def test_reid_and_wrong_botsort_gmc_are_rejected(self) -> None:
        reid_data = self.config_data()
        reid_data["trackers"][1]["reid"] = True  # type: ignore[index]
        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(reid_data)

        gmc_data = self.config_data()
        gmc_data["trackers"][1]["gmc_method"] = "none"  # type: ignore[index]
        with self.assertRaises(ValidationError):
            M1ExperimentConfig.model_validate(gmc_data)

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

    def test_draft_is_inspectable_but_not_execution_ready(self) -> None:
        readiness = self.config.execution_readiness()

        self.assertEqual(readiness.status, ReadinessStatus.BLOCKED)
        self.assertIn("runtime.ultralytics_version", readiness.unresolved_fields)
        self.assertIn("runtime.torch_version", readiness.unresolved_fields)
        self.assertIn("runtime.cuda_version", readiness.unresolved_fields)
        self.assertIn("runtime.hardware_confirmed", readiness.unresolved_fields)
        self.assertIn("detectors.yolo11n.weights_sha256", readiness.unresolved_fields)
        self.assertIn("detectors.yolo26n.weights_sha256", readiness.unresolved_fields)
        with self.assertRaises(ValueError):
            self.config.require_execution_ready()

    def test_synthetic_confirmed_provenance_becomes_execution_ready(self) -> None:
        data = self.config_data()
        data["runtime"].update(  # type: ignore[union-attr]
            {
                "ultralytics_version": "synthetic-ultralytics-version",
                "torch_version": "synthetic-torch-version",
                "cuda_version": "synthetic-cuda-version",
                "hardware_confirmed": True,
            }
        )
        data["detectors"][0]["weights_sha256"] = "a" * 64  # type: ignore[index]
        data["detectors"][1]["weights_sha256"] = "b" * 64  # type: ignore[index]
        ready_config = M1ExperimentConfig.model_validate(data)

        self.assertTrue(ready_config.execution_readiness().ready)
        ready_config.require_execution_ready()

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
        self.assertEqual(self.config.measurement.warmup_frames, 30)
        self.assertTrue(
            self.config.measurement.fresh_detector_pass_required_for_publishable_run
        )
        self.assertTrue(self.config.measurement.same_run_resume_cache_allowed)
        self.assertTrue(self.config.measurement.cross_run_cache_allowed_for_development)
        self.assertFalse(self.config.measurement.cross_run_cache_publishable)


if __name__ == "__main__":
    unittest.main()
