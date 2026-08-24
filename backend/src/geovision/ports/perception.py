"""Framework-independent interfaces for perception components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from geovision.domain.models import Detection, FrameRef, TrackObservation


class Detector(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, frame: Any, frame_ref: FrameRef) -> Sequence[Detection]: ...


class Tracker(Protocol):
    @property
    def name(self) -> str: ...

    def update(
        self,
        detections: Sequence[Detection],
        frame: Any,
        frame_ref: FrameRef,
    ) -> Sequence[TrackObservation]: ...

    def reset(self) -> None: ...
