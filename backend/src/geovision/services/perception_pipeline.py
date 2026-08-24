"""Small orchestration layer joining detector and tracker ports."""

from __future__ import annotations

from typing import Any

from geovision.domain.models import FrameRef, PipelineResult
from geovision.ports.perception import Detector, Tracker


class PerceptionPipeline:
    def __init__(self, detector: Detector, tracker: Tracker) -> None:
        self._detector = detector
        self._tracker = tracker

    @property
    def detector_name(self) -> str:
        return self._detector.name

    @property
    def tracker_name(self) -> str:
        return self._tracker.name

    def process(self, frame: Any, frame_ref: FrameRef) -> PipelineResult:
        detections = tuple(self._detector.detect(frame, frame_ref))
        tracks = tuple(self._tracker.update(detections, frame, frame_ref))
        return PipelineResult(frame=frame_ref, detections=detections, tracks=tracks)

    def reset(self) -> None:
        self._tracker.reset()

