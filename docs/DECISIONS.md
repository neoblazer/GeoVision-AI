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

## ADR-009: Dual mission scope

GeoVision supports two mission modes: `border_surveillance` and
`search_and_rescue`. Both use the same evidence and provenance foundations but
have separately defined rules, evaluation data, and operator interpretation.
Mission context is established before perception processing so observations,
entities, events, and alerts cannot silently cross mission boundaries.

The primary novelty is **Persistent Mission-Oriented Intelligence with
evidence-backed event reasoning for resource-constrained UAV border
surveillance and search-and-rescue missions.**

## ADR-010: Non-lethal human-in-the-loop operation

GeoVision is a non-lethal decision-support and evidence-reporting platform. It
may produce reason-coded mission events and candidate alerts, but an authorized
human reviews, interprets, acknowledges, dismisses, or escalates them. The
system never makes an autonomous hostile classification. GeoVision AI must
never autonomously recommend, select, or execute force or lethal action. Human
review of evidence does not authorize the system to recommend force.

## ADR-011: Mission authorization semantics

Mission authorization has three states: `authorized`, `unauthorized`, and
`unknown`; the default is `unknown`. The state may change only through an
authenticated mission input, documented access-control input, or explicit
authorized-operator action, with scope and provenance retained.

Mission authorization status, ally/enemy status, hostility, nationality, and
intent are never inferred from a face, clothing, ethnicity, religion, gender,
another protected characteristic, or general appearance. Behavior may
contribute to a documented event such as loitering or zone crossing, but it
must not determine nationality, identity, ally/enemy status, or authorization.

`unauthorized` means that permission for the relevant mission or zone is absent
or not established; it does not mean enemy or hostile and does not authorize
force.

## ADR-012: Persistent entities are operational hypotheses

A Persistent Mission Entity is a mission-scoped, evidence-backed hypothesis
that observations belong to one continuing operational entity. It is not a
biometric identity, civil identity, or confirmed real-world identity.

Transient tracker identifiers are eventually represented by source-qualified
and tracker-instance-qualified TrackKeys. Mission Memory owns persistent entity
identifiers and must preserve reconciliation evidence, uncertainty, reason
codes, and provenance rather than promoting a bare tracker ID to an identity.

## ADR-013: Evidence-backed alerts and operator responsibility

Deterministic mission events, immutable evidence packets, and operator alerts
have distinct lifecycles. Evidence packets reference observations, entities,
rules, reason codes, uncertainty, and model/configuration provenance. Detection
confidence and event confidence remain separate, and confidence alone is
insufficient to issue or justify an alert.

Candidate alerts require human review. Alert transitions record the authorized
actor, timestamp, reason, and evidence version without rewriting the underlying
event. Operational interpretation and response remain the operator's
responsibility.

## ADR-014: Visible-weapon cues are separately gated

A visible-weapon cue detector is a planned, separately gated final-product
capability. It must be independently selected, implemented, and validated before
operational use. The current COCO YOLO11n and YOLO26n baselines must not be
described as firearm detectors.

A visible-weapon cue is evidence for operator review, not autonomous
confirmation of hostility or intent. Evaluation requires suitable held-out
public data, precision/recall reporting, and mission-relevant failure analysis
before integration. Concealed-weapon claims and automatic hostile or force
recommendations are prohibited.

## ADR-015: Metric distance and location fail unavailable

The existing `DepthService -> DistanceService -> FusionService` separation is
preserved, and backend selection remains separately gated. Metric distance or
location may be emitted only when calibration, scale, geometry, reference frame,
and provenance establish its validity.

When metric validity cannot be established, the result is explicitly
`unavailable` with a machine-readable reason and uncertainty metadata. Relative
depth, image coordinates, stale estimates, or assumptions must not be presented
as validated metric distance or location.

## ADR-016: Milestone 1 recorded-file execution semantics

- Both detectors share one comparison policy: detection task, image size 640,
  confidence floor 0.10, IoU 0.70, maximum 300 detections, rectangular
  inference, all classes, class-aware processing, no augmentation, model-native
  end-to-end behavior, no compilation, and no channels-last conversion. Batch
  size 1, FP16, CUDA device 0, and 30 warm-up frames are fixed separately as
  runtime and measurement semantics.
- The former 0.35 confidence threshold is replaced by 0.10 because 0.35 removed
  observations from ByteTrack's configured low-confidence recovery region
  before the tracker could evaluate them. The 0.10 value is a detector-cache
  input floor, not a reporting or evidence-acceptance threshold. Reporting and
  evaluation thresholds remain separate and unresolved.
- YOLO11 and YOLO26 retain their model-native postprocessing provenance.
  YOLO26's end-to-end path differs internally from YOLO11's ordinary NMS path,
  but both must emit the same GeoVision detection contract. The configured IoU
  value is always supplied and recorded even when an end-to-end path does not
  apply ordinary NMS IoU identically.
- ByteTrack and BoT-SORT both use `track_high_thresh: 0.25`,
  `track_low_thresh: 0.10`, `new_track_thresh: 0.25`, `track_buffer: 30`,
  `match_thresh: 0.80`, and `fuse_score: true`. `track_buffer` is measured
  directly in frames without frame-rate scaling. ByteTrack has no GMC or ReID.
  BoT-SORT requires native `sparseOptFlow` GMC, disables ReID, and retains
  `proximity_thresh: 0.50`, `appearance_thresh: 0.80`, and `model: auto` as
  explicit installed-API inputs. Automatic dependency and model installation
  are prohibited.
- Comparative recorded files use zero-based contiguous frame indices and
  canonical time `frame_index / validated_fps`. FPS must be finite and positive
  before inference. Source timestamps are diagnostic; unavailable, duplicated,
  or regressive values never replace canonical time and their status must be
  recorded by the future source implementation. Decode is sequential, with no
  seeking, skipping, or retrying. A reported frame count is valid only when it
  is a non-negative integer. EOF is valid only when exactly that many frames
  decode successfully and the next read fails. A failed read before the count,
  any count mismatch or extra decoded frame, and missing count metadata all
  fail the run; missing count makes termination ambiguous.
- Decoded frames must be BGR `uint8` H x W x 3, with decoded dimensions taking
  precedence over container metadata. Future execution fingerprints the source
  file by SHA-256 and byte size and each replayed decoded frame by SHA-256.
  Research artifacts never contain source paths.
- Persisted numeric values fail closed on NaN or infinity. Bounding boxes also
  require positive area. Because frame dimensions are unavailable in the box
  domain object, future adapters must reject out-of-frame boxes at their
  boundary rather than silently clamp them. Invalid metric distance remains
  unavailable and is never converted into relative depth.
- Canonical detections sort by descending confidence, then ascending class ID,
  `x1`, `y1`, `x2`, `y2`, and original detection ordinal. Adapters convert
  NumPy or Torch scalars to Python numeric values. Canonical JSON preserves full
  finite precision without arbitrary rounding, normalizes negative zero to
  `0.0`, and uses Python 3.11 shortest-round-trip floats, UTF-8, sorted keys,
  compact separators, and `allow_nan=False`. These rules and immutable artifact
  schema identifiers are part of experiment provenance, not benchmark results.
  The experiment manifest is the sole canonical manifest artifact and retains
  `geovision.experiment-manifest/v1`; “run manifest” is only a descriptive
  synonym, not a second artifact or schema.

## ADR-017: Deferred runtime dependency and licensing reviews

If Phase 1D-D production code imports `psutil`, that package must become an
explicit direct vision/runtime dependency in that phase. It is not added as a
dependency by this decision-only phase.

Ultralytics package and checkpoint licensing, together with publication and
distribution obligations for research artifacts, require a separate review of
official sources before distribution. This ADR records the review gate and
does not assert a licensing conclusion.
