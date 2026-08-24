# Implementation Milestones

## M0 - Clean foundation (complete)

- Repository hygiene and dependency boundaries.
- Immutable domain schemas.
- Detector/tracker ports and deterministic mocks.
- Minimal pipeline, API application factory, and tests.
- Baseline and candidate experiment configurations.

## M1 - Reproducible perception baseline

- Webcam, file, and RTSP frame sources.
- YOLO11n and YOLO26n adapters.
- ByteTrack and BoT-SORT configurations.
- Per-frame latency, FPS, dropped-frame, RAM, and VRAM logging.
- MOT-format result export and cached detector outputs.

Exit criterion: the same command can run all four detector/tracker configurations
and produce comparable machine-readable benchmark results.

## M2 - Camera-motion service

- Background masking using detector boxes.
- Sparse optical-flow correspondences.
- RANSAC transformation estimation.
- Inlier ratio, residual error, and reliability output.
- Keyframe-relative stabilized footpoints.

Exit criterion: camera-motion transformations and confidence are replayable from
stored logs and can be enabled or disabled as an ablation.

## M3 - Persistent Mission Memory

- Tracklet lifecycle and immutable association evidence.
- Reliability-aware cue fusion.
- Hard feasibility gates and one-to-one assignment.
- Persistent Mission Entities and last-known location.

Exit criterion: synthetic fragmentation tests and held-out public sequences show
measurable reconciliation outcomes without silent entity merges.

## M4 - Event Engine and evaluation

- Restricted-zone and loitering rules.
- Temporal hysteresis and provenance.
- Standard MOT metrics and mission-centric metrics.
- Baselines and component ablations.

Exit criterion: one reproducible experiment table can be generated from stored
configuration and result files.

## M5 - Operator platform

- Live dashboard, replay, analytics, and report export.
- Depth and segmentation as scheduled optional services.
- Structured-data-only LLM summary.

Exit criterion: the demo consumes the same stored Mission Memory and Event Engine
records used for research evaluation.

