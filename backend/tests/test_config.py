from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from geovision.core.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_research_safe(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()

        self.assertEqual(settings.detector_backend, "mock")
        self.assertEqual(settings.detector_model, "yolo11n.pt")
        self.assertEqual(settings.detector_candidate_model, "yolo26n.pt")
        self.assertEqual(settings.tracker_backend, "botsort")
        self.assertFalse(settings.tracker_reid)
        self.assertFalse(settings.depth_enabled)
        self.assertFalse(settings.segmentation_enabled)

    def test_boolean_environment_validation(self) -> None:
        with patch.dict(os.environ, {"GEOVISION_TRACKER_REID": "sometimes"}, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_environment()


if __name__ == "__main__":
    unittest.main()

