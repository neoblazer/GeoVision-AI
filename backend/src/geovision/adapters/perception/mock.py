"""Deterministic test doubles for the perception pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from geovision.domain.models import BoundingBox, Detection, FrameRef, TrackObservation


class MockDetector:
    @property
    def name(self) -> str:
        return "mock-detector"

    def detect(self, frame: Any, frame_ref: FrameRef) -> Sequence[Detection]:
        del frame
        return (
            Detection(
                detection_id=f"{frame_ref.source_id}:{frame_ref.frame_id}:0",
                frame=frame_ref,
                class_id=0,
                label="person",
                confidence=0.90,
                bbox=BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=100.0),
            ),
        )


class MockTracker:
    def __init__(self) -> None:
        self._track_id = 1

    @property
    def name(self) -> str:
        return "mock-tracker"

    def update(
        self,
        detections: Sequence[Detection],
        frame: Any,
        frame_ref: FrameRef,
    ) -> Sequence[TrackObservation]:
        del frame, frame_ref
        return tuple(
            TrackObservation(
                track_id=self._track_id + index,
                frame=detection.frame,
                class_id=detection.class_id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
            for index, detection in enumerate(detections)
        )

    def reset(self) -> None:
        self._track_id = 1
