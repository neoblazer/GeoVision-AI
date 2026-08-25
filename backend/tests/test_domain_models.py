from __future__ import annotations

import unittest
from math import isfinite, isinf

from pydantic import ValidationError

from geovision.domain.enums import AssociationOutcome, DistanceSource, EntityState, EventType
from geovision.domain.models import (
    AssociationDecision,
    BoundingBox,
    Detection,
    DistanceEstimate,
    FrameRef,
    MissionEntity,
    MissionEvent,
    MotionEstimate,
    TrackObservation,
)


class DomainModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameRef(source_id="test", frame_id=1, timestamp_s=0.04)

    def test_bbox_exposes_tracking_footpoint(self) -> None:
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=80)
        self.assertEqual(bbox.width, 20)
        self.assertEqual(bbox.height, 60)
        self.assertEqual(bbox.footpoint, (20, 80))

    def test_large_bbox_uses_overflow_safe_midpoint(self) -> None:
        bbox = BoundingBox(x1=1e308, y1=10.0, x2=1.1e308, y2=30.0)

        self.assertGreater(bbox.width, 0.0)
        self.assertTrue(isfinite(bbox.width))
        self.assertTrue(isinf((bbox.x1 + bbox.x2) / 2.0))
        self.assertTrue(isfinite(bbox.footpoint[0]))
        self.assertAlmostEqual(bbox.footpoint[0] / 1.05e308, 1.0, places=15)

    def test_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            BoundingBox(x1=30, y1=20, x2=10, y2=80)

        for coordinates in (
            (10.0, 20.0, 10.0, 80.0),
            (10.0, 20.0, 30.0, 20.0),
            (float("nan"), 20.0, 30.0, 80.0),
            (10.0, float("inf"), 30.0, 80.0),
            (10.0, 20.0, float("-inf"), 80.0),
            (-1e308, 20.0, 1e308, 80.0),
        ):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValidationError):
                BoundingBox(
                    x1=coordinates[0],
                    y1=coordinates[1],
                    x2=coordinates[2],
                    y2=coordinates[3],
                )

    def test_non_finite_timestamp_is_rejected(self) -> None:
        for timestamp in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValidationError):
                FrameRef(source_id="test", frame_id=1, timestamp_s=timestamp)

    def test_non_finite_detection_and_track_confidence_are_rejected(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=80.0)

        for confidence in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(model="detection", confidence=confidence):
                with self.assertRaises(ValidationError):
                    Detection(
                        detection_id="detection-1",
                        frame=self.frame,
                        class_id=0,
                        label="person",
                        confidence=confidence,
                        bbox=bbox,
                    )
            with self.subTest(model="track", confidence=confidence):
                with self.assertRaises(ValidationError):
                    TrackObservation(
                        track_id=1,
                        frame=self.frame,
                        class_id=0,
                        label="person",
                        confidence=confidence,
                        bbox=bbox,
                    )

    def test_detection_and_track_confidence_accept_boundaries(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=80.0)

        for confidence in (0.0, 1.0):
            with self.subTest(confidence=confidence):
                detection = Detection(
                    detection_id=f"detection-{confidence}",
                    frame=self.frame,
                    class_id=0,
                    label="person",
                    confidence=confidence,
                    bbox=bbox,
                )
                track = TrackObservation(
                    track_id=1,
                    frame=self.frame,
                    class_id=0,
                    label="person",
                    confidence=confidence,
                    bbox=bbox,
                )

                self.assertEqual(detection.confidence, confidence)
                self.assertEqual(track.confidence, confidence)

    def test_non_finite_motion_values_are_rejected(self) -> None:
        valid = {
            "frame": self.frame,
            "homography": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "confidence": 0.9,
            "inlier_ratio": 0.8,
            "residual_px": 1.5,
            "reliable": True,
        }
        invalid_cases = (
            {"homography": (float("nan"), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)},
            {"confidence": float("inf")},
            {"inlier_ratio": float("-inf")},
            {"residual_px": float("nan")},
        )

        for changes in invalid_cases:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                MotionEstimate(**(valid | changes))

    def test_non_finite_available_distance_is_rejected(self) -> None:
        for field_name, value in (
            ("distance_m", float("nan")),
            ("distance_m", float("inf")),
            ("confidence", float("-inf")),
        ):
            values = {
                "track_id": 1,
                "frame": self.frame,
                "distance_m": 5.0,
                "source": DistanceSource.MONOCULAR,
                "confidence": 0.5,
            }
            values[field_name] = value
            with self.subTest(field=field_name, value=value), self.assertRaises(
                ValidationError
            ):
                DistanceEstimate(**values)

    def test_unavailable_distance_cannot_claim_metres(self) -> None:
        with self.assertRaises(ValidationError):
            DistanceEstimate(
                track_id=1,
                frame=self.frame,
                distance_m=5.0,
                source=DistanceSource.UNAVAILABLE,
                confidence=0.0,
            )

    def test_available_metric_distance_remains_valid(self) -> None:
        distance = DistanceEstimate(
            track_id=1,
            frame=self.frame,
            distance_m=5.0,
            source=DistanceSource.GEOMETRIC,
            confidence=1.0,
            calibrated=True,
        )

        self.assertEqual(distance.distance_m, 5.0)
        self.assertEqual(distance.source, DistanceSource.GEOMETRIC)
        self.assertTrue(distance.calibrated)

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

    def test_remaining_persisted_float_fields_reject_non_finite_values(self) -> None:
        entity_values = {
            "entity_id": "entity-1",
            "track_ids": (1,),
            "class_id": 0,
            "label": "person",
            "first_seen_s": 1.0,
            "last_seen_s": 2.0,
            "observation_count": 2,
            "state": EntityState.ACTIVE,
            "last_known_footpoint": (20.0, 80.0),
        }
        for changes in (
            {"first_seen_s": float("nan")},
            {"last_seen_s": float("inf")},
            {"last_known_footpoint": (20.0, float("-inf"))},
        ):
            with self.subTest(model="entity", changes=changes):
                with self.assertRaises(ValidationError):
                    MissionEntity(**(entity_values | changes))

        association_values = {
            "candidate_track_id": 1,
            "entity_id": "entity-1",
            "outcome": AssociationOutcome.MERGED,
            "total_score": 0.9,
            "motion_score": 0.8,
            "appearance_score": None,
            "scale_score": 0.7,
            "time_score": 0.6,
            "motion_reliability": 0.9,
            "appearance_reliability": 0.0,
            "reasons": ("consistent-motion",),
        }
        for field_name in (
            "total_score",
            "motion_score",
            "appearance_score",
            "scale_score",
            "time_score",
            "motion_reliability",
            "appearance_reliability",
        ):
            with self.subTest(model="association", field=field_name):
                with self.assertRaises(ValidationError):
                    AssociationDecision(**(association_values | {field_name: float("nan")}))

        event_values = {
            "event_id": "event-1",
            "event_type": EventType.ZONE_ENTRY,
            "entity_id": "entity-1",
            "started_at_s": 1.0,
            "confirmed_at_s": 2.0,
            "confidence": 0.8,
            "rule_id": "rule-1",
            "evidence_frame_ids": (1,),
        }
        for field_name in ("started_at_s", "confirmed_at_s", "confidence"):
            with self.subTest(model="event", field=field_name):
                with self.assertRaises(ValidationError):
                    MissionEvent(**(event_values | {field_name: float("inf")}))

    def test_valid_float_bearing_models_still_parse(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=80.0)
        detection = Detection(
            detection_id="detection-1",
            frame=self.frame,
            class_id=0,
            label="person",
            confidence=0.9,
            bbox=bbox,
        )
        track = TrackObservation(
            track_id=1,
            frame=self.frame,
            class_id=0,
            label="person",
            confidence=0.8,
            bbox=bbox,
        )
        motion = MotionEstimate(
            frame=self.frame,
            homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            confidence=0.9,
            inlier_ratio=0.8,
            residual_px=1.5,
            reliable=True,
        )
        unavailable_distance = DistanceEstimate(
            track_id=1,
            frame=self.frame,
            distance_m=None,
            source=DistanceSource.UNAVAILABLE,
            confidence=0.0,
        )

        self.assertEqual(detection.bbox, bbox)
        self.assertEqual(track.frame, self.frame)
        self.assertTrue(motion.reliable)
        self.assertIsNone(unavailable_distance.distance_m)


if __name__ == "__main__":
    unittest.main()
