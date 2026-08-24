# Research Protocol

## Primary hypothesis

A camera-motion-aware, reliability-adaptive Mission Memory reduces duplicate
entity reports and false mission alerts under camera motion and track
fragmentation while retaining practical throughput on a 4 GB laptop GPU.

## Required baselines

1. YOLO11n + ByteTrack.
2. YOLO11n + BoT-SORT.
3. YOLO26n + ByteTrack.
4. YOLO26n + BoT-SORT.
5. Best detector/tracker + naive fixed-gap tracklet merging.
6. Best detector/tracker + proposed Mission Memory.

Detector outputs should be cached for tracker comparisons so the tracker is the
only changed independent variable.

## Ablations

- Without camera-motion compensation.
- Fixed cue weights instead of reliability-aware weights.
- Without appearance cue.
- Without event hysteresis.
- Without long-term reconciliation.

## Standard metrics

- HOTA and AssA.
- IDF1.
- ID switches.
- Fragmentations.
- Detection precision and recall.
- Mean, median, and p95 frame latency.
- FPS, dropped-frame rate, peak RAM, and peak VRAM.

## Mission metrics

- Duplicate Report Rate.
- False Merge Rate.
- Missed Entity Rate.
- Correct Reacquisition Rate.
- Entity Count Error.
- Event precision, recall, and F1.
- False alerts per minute during camera-motion segments.
- Alert latency.

## Data

Use public, identity-annotated UAV sequences. Tune thresholds on development
sequences and report final results on held-out sequences. Derived event labels
must be versioned separately from model inputs and must never be used for model
training.

## Milestone 1 perception protocol

The four required detector/tracker results are generated from two detector
passes over an immutable recorded file. YOLO11n runs once and its normalized
detections are replayed through ByteTrack and BoT-SORT. YOLO26n then follows the
same procedure. BoT-SORT uses native `sparseOptFlow` GMC and ReID remains
disabled. Webcam and RTSP sources are live-smoke inputs, not comparative matrix
sources.

YAML owns all experiment-affecting settings. Application `Settings` do not
configure benchmark experiments, and future CLI or environment inputs are
limited to operational paths and secrets. The resolved configuration must be
immutable, reject unknown fields, and serialize deterministically.

The first 30 processed frames are warm-up frames by default; the value remains
explicit and configurable. A new publishable performance run executes each
detector and creates fresh caches bound to that run. An interrupted run may
resume its own cache. A cache from another run may be used for development, but
the resulting run is non-publishable unless a fresh detector pass and detector
timings belong to the current run.

Detection caches use versioned JSONL and experiment manifests use versioned
JSON. MOT export uses one-based frame numbers, positive run-local remapped track
IDs, and one-based top-left coordinates. Exact dependency versions, model
weight hashes, and hardware settings must be explicit before execution
readiness is granted; unresolved draft configurations remain inspectable but
cannot authorize a benchmark run. The public UAV dataset must be selected
before publishable accuracy evaluation, but does not block local runner work.

Mission Memory, persistent reconciliation, events, standalone camera-motion
estimation, depth, distance, segmentation, dashboard, replay, scheduling, and
LLM reporting remain outside Milestone 1.

## Milestone 1B verified vision runtime

The local compatibility target is an NVIDIA GeForce RTX 3050 Laptop GPU with
4 GiB VRAM and compute capability 8.6. Torch reported CUDA availability, and a
small FP16 CUDA matrix multiplication passed. Its peak allocated CUDA memory
was 8,912,896 bytes. That measurement is neither detector memory usage nor a
performance benchmark and must not be reported as either.

ByteTrack and BoT-SORT both consumed externally supplied deterministic
detections over two synthetic frames. BoT-SORT used native `sparseOptFlow` GMC
with ReID disabled. No model or dataset was used in this compatibility
validation.

`hardware_confirmed` remains false until YOLO11n and YOLO26n both pass
controlled 640 x 640 FP16 smoke tests on the target GPU. Their model-weight
hashes also remain unresolved, so benchmark execution continues to fail closed.

## Milestone 1C YOLO11n checkpoint compatibility evidence

The verified `yolo11n.pt` checkpoint came from the official Ultralytics v8.3.0
asset at
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`.
Its canonical SHA-256 is
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`,
and its exact size is 5,613,764 bytes. The checkpoint remains outside Git under
`%LOCALAPPDATA%\GeoVision-AI\models\ultralytics\v8.3.0`.

Ultralytics 8.4.127 loaded the `yolo11n.yaml` detection model with 2,624,080
parameters and 80 COCO classes using Torch 2.13.0+cu130, TorchVision
0.28.0+cu130, and the CUDA 13.0 runtime. A batch-one, 640 x 640 synthetic-input
FP16 CUDA compatibility smoke completed three warm-up calls and five smoke
calls without CUDA OOM or CPU fallback. Peak allocated memory was 51,166,720
bytes and peak reserved memory was 69,206,016 bytes. Zero detections were
expected because the synthetic input contained no scene objects.

This was a compatibility test, not an accuracy evaluation or publishable
performance benchmark. Hardware remains unconfirmed until YOLO26n also passes
its controlled smoke test.
