from __future__ import annotations

import unittest

from pydantic import ValidationError

from geovision.domain.models import (
    BoundingBox,
    Detection,
    FrameRef,
    PipelineResult,
    TrackObservation,
)


class PipelineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameRef(source_id="source", frame_id=4, timestamp_s=0.16)
        self.other_frame = FrameRef(source_id="source", frame_id=5, timestamp_s=0.20)
        self.bbox = BoundingBox(x1=1.0, y1=2.0, x2=11.0, y2=22.0)

    def detection(self, frame: FrameRef) -> Detection:
        return Detection(
            detection_id=f"{frame.source_id}:{frame.frame_id}:0",
            frame=frame,
            class_id=0,
            label="person",
            confidence=0.9,
            bbox=self.bbox,
        )

    def track(self, frame: FrameRef) -> TrackObservation:
        return TrackObservation(
            track_id=1,
            frame=frame,
            class_id=0,
            label="person",
            confidence=0.9,
            bbox=self.bbox,
        )

    def test_matching_frame_references_are_accepted(self) -> None:
        result = PipelineResult(
            frame=self.frame,
            detections=(self.detection(self.frame),),
            tracks=(self.track(self.frame),),
        )

        self.assertEqual(result.frame, self.frame)
        self.assertEqual(set(PipelineResult.model_fields), {"frame", "detections", "tracks"})

    def test_mismatched_detection_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "every detection must reference"):
            PipelineResult(
                frame=self.frame,
                detections=(self.detection(self.other_frame),),
                tracks=(),
            )

    def test_mismatched_track_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "every track must reference"):
            PipelineResult(
                frame=self.frame,
                detections=(),
                tracks=(self.track(self.other_frame),),
            )


if __name__ == "__main__":
    unittest.main()
