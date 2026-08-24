from __future__ import annotations

import unittest

from pydantic import ValidationError

from geovision.domain.enums import DistanceSource, EntityState
from geovision.domain.models import BoundingBox, DistanceEstimate, FrameRef, MissionEntity


class DomainModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameRef(source_id="test", frame_id=1, timestamp_s=0.04)

    def test_bbox_exposes_tracking_footpoint(self) -> None:
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=80)
        self.assertEqual(bbox.width, 20)
        self.assertEqual(bbox.height, 60)
        self.assertEqual(bbox.footpoint, (20, 80))

    def test_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BoundingBox(x1=30, y1=20, x2=10, y2=80)

    def test_unavailable_distance_cannot_claim_metres(self) -> None:
        with self.assertRaises(ValidationError):
            DistanceEstimate(
                track_id=1,
                frame=self.frame,
                distance_m=5.0,
                source=DistanceSource.UNAVAILABLE,
                confidence=0.0,
            )

    def test_entity_requires_unique_track_ids(self) -> None:
        with self.assertRaises(ValidationError):
            MissionEntity(
                entity_id="entity-1",
                track_ids=(1, 1),
                class_id=0,
                label="person",
                first_seen_s=1.0,
                last_seen_s=2.0,
                observation_count=2,
                state=EntityState.ACTIVE,
                last_known_footpoint=(20.0, 80.0),
            )


if __name__ == "__main__":
    unittest.main()

