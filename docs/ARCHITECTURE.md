# Architecture

## Design rules

1. Domain schemas do not import FastAPI, OpenCV, Ultralytics, database clients,
   or frontend code.
2. Model implementations live behind ports in `adapters/`.
3. Raw tracker IDs and persistent mission entity IDs are different concepts.
4. Every frame carries an immutable source ID, frame number, and timestamp.
5. Missing or unvalidated metric distance is represented explicitly; it is never
   silently substituted with relative depth.
6. Every entity reconciliation and mission alert records machine-readable
   evidence.
7. Experimental code lives outside production packages and is never vendored.

## Target data flow

```text
Input service
  -> frame packet
  -> detector adapter
  -> tracker adapter
  -> camera-motion service
  -> reliability-aware reconciliation
  -> persistent Mission Memory
  -> deterministic Event Engine
  -> API / replay / analytics / structured report
```

Depth, distance, semantic segmentation, and scheduling are parallel optional
services. Their failures must not stop the core detection/tracking pipeline.

## Stable identifiers

- `source_id`: one input stream or recorded video.
- `frame_id`: monotonically increasing within one source.
- `track_id`: transient identifier owned by the selected tracker.
- `entity_id`: mission-scoped persistent identifier owned by Mission Memory.
- `event_id`: immutable event record with rule and evidence provenance.

