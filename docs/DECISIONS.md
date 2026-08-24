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

## ADR-006: Milestone 1 reproducible perception protocol

- YAML is the sole owner of benchmark experiment semantics. CLI arguments and
  environment variables are reserved for operational paths and secrets, and
  application `Settings` do not configure benchmark experiments.
- The comparative matrix accepts immutable recorded files only. Webcam and RTSP
  remain non-comparative ingestion and live-smoke source types.
- YOLO11n and YOLO26n each execute once per recorded source. Their normalized
  detections are replayed through both ByteTrack and BoT-SORT.
- ByteTrack is the baseline. BoT-SORT uses native `sparseOptFlow` GMC with ReID
  disabled. The standalone `CameraMotionService` remains outside Milestone 1.
- A future tracker adapter will isolate one exactly pinned Ultralytics internal
  API. Milestone 1A defines contracts only and does not select or integrate it.
- Publishable performance runs require a fresh detector pass and fresh,
  run-bound caches. Same-run resume is allowed. Cross-run cache reuse is for
  development only and makes a run non-publishable unless fresh detector
  timings from the current run are present.
- Detection caches use versioned JSONL and manifests use versioned JSON. MOT
  export uses one-based frames, positive run-local remapped track IDs, and
  one-based top-left coordinates.
- CUDA device, batch size, image size, FP16, dependency versions, model-weight
  hashes, and the public research dataset remain provisional until validated.

## ADR-007: Depth backend clarification

Depth Anything V2 Metric VKITTI ViT-S is the optional primary depth backend.
Metric3D may exist only as a small standalone experiment and is not integrated
into the live pipeline without separate approval. MoGe-2 remains excluded from
live use after measuring approximately 400-1200+ ms per frame on the target RTX
3050 Laptop GPU. Depth, distance, and segmentation are outside Milestone 1.

## ADR-008: Milestone 1 Windows vision runtime

- The verified Windows and Python 3.11 vision stack is Torch 2.13.0+cu130,
  TorchVision 0.28.0+cu130, the CUDA 13.0 runtime bundled with the PyTorch
  wheel, Ultralytics 8.4.127, LAP 0.5.12, and OpenCV 4.14.0.94.
- The CUDA 13.0 runtime is supplied by the PyTorch wheel. GeoVision does not
  require a separately installed CUDA Toolkit for this runtime.
- Regular `opencv-python` replaces `opencv-python-headless` because the pinned
  Ultralytics distribution explicitly requires regular OpenCV. The two OpenCV
  distributions must never coexist in one environment.
- Future GeoVision tracker adapters isolate the pinned Ultralytics tracker
  internals because those internal APIs are version-sensitive.
- No depth-estimation dependency belongs to Milestone 1.
