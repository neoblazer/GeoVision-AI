from __future__ import annotations

import unittest

from geovision.adapters.perception import MockDetector, MockTracker
from geovision.domain.models import FrameRef
from geovision.services.perception_pipeline import PerceptionPipeline


class PerceptionPipelineTests(unittest.TestCase):
    def test_mock_pipeline_preserves_frame_identity(self) -> None:
        pipeline = PerceptionPipeline(MockDetector(), MockTracker())
        frame_ref = FrameRef(source_id="fixture", frame_id=7, timestamp_s=0.28)

        result = pipeline.process(frame=object(), frame_ref=frame_ref)

        self.assertEqual(result.frame, frame_ref)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].frame, frame_ref)
        self.assertEqual(result.tracks[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()

