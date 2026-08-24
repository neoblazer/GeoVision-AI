# Architectural Decisions

## ADR-001: Research contribution

The headline contribution is motion-aware persistent mission intelligence:
reconciling fragmented UAV tracklets into explainable Mission Entities and
improving entity-level reporting and event accuracy.

The detector, depth model, segmentation model, scheduler, dashboard, replay,
and LLM summary are supporting components.

## ADR-002: Detector strategy

- Reproducible baseline: YOLO11n.
- Candidate: YOLO26n.
- No replacement is accepted using vendor COCO numbers alone. Both models must
  be tested on identical UAV clips and the target RTX 3050.

## ADR-003: Tracker strategy

- Research baseline: ByteTrack.
- Moving-camera production candidate: BoT-SORT with sparse optical-flow camera
  motion compensation and ReID disabled initially.
- ReID is enabled only after a controlled ablation demonstrates useful identity
  gains relative to its compute cost.

## ADR-004: Depth and distance

Keep the `DepthService -> DistanceService -> FusionService` separation. Depth
Anything V2 Metric VKITTI ViT-S remains an optional auxiliary backend. MoGe-2
and Depth Pro are excluded from the live pipeline because of measured latency
and memory failures on the target hardware.

Distance results must record source, confidence, calibration status, and a
warning when metric validity is not established.

## ADR-005: Repository hygiene

Do not commit virtual environments, model weights, datasets, generated output,
vendor repositories, notebooks used only for one-off experiments, caches, or
duplicate legacy modules.

