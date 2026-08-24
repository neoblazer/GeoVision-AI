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

