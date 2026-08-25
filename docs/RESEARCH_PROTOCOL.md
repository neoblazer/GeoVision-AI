# Research Protocol

## Finalized research question

> On a resource-constrained RGB UAV platform, does mission-scoped persistent
> entity reconciliation combined with a deterministic evidence-backed Event
> Engine improve entity continuity, suppress duplicate reports, and produce more
> traceable border-surveillance and SAR alerts than ordinary frame-level or
> transient-track reporting, without exceeding practical latency and memory
> limits?

The primary novelty is **Persistent Mission-Oriented Intelligence with
evidence-backed event reasoning for resource-constrained UAV border
surveillance and search-and-rescue missions.** Persistent Mission Entities,
Mission Memory, and the deterministic Event Engine form the primary research
contribution. Adaptive model or frame scheduling is only a supporting resource
optimization.

The earlier hypothesis framing was that a camera-motion-aware,
reliability-adaptive Mission Memory reduces duplicate entity reports and false
mission alerts under camera motion and track fragmentation while retaining
practical throughput on a 4 GB laptop GPU. The finalized question expands this
historical framing to both mission modes, evidence traceability, and an explicit
Event Engine comparison.

## Proposed hypotheses

These hypotheses are not pre-registered or locked. They become locked only
after metric definitions, acceptance thresholds, dataset versions, split
manifests, exclusion rules, and statistical procedures are finalized before
evaluated test results are inspected. No threshold is implied by this draft.

- **H1 - Entity continuity:** mission-scoped reconciliation improves
  Unique-Entity Recall and reconciliation/continuity accuracy relative to
  transient-track reporting while keeping False Merge Rate within a declared
  acceptance bound.
- **H2 - Duplicate suppression:** entity-based reporting reduces Duplicate
  Report Rate and duplicate-alert rate without increasing missed-event rate.
- **H3 - Event quality:** deterministic, mission-specific reasoning reduces
  false alerts and duplicate alerts while preserving or improving event recall
  compared with frame-level or transient-track rules.
- **H4 - Evidence quality:** reason-coded evidence packets improve evidence
  completeness and end-to-end traceability from an alert to its source frames,
  observations, entity associations, rule version, and configuration/model
  provenance.
- **H5 - Operational latency:** persistence and event reasoning retain practical
  time-to-alert and do not exceed the declared latency, RAM, or VRAM limits of
  the target laptop.
- **H6 - SAR continuity:** mission-scoped entities improve continuity of a SAR
  subject's last-known valid location and suppress duplicate-victim reports
  without treating a possible-distress cue as a medical diagnosis.

## Required baselines

These are required for the completed research evaluation; the Mission Memory
baselines and real perception runs are planned and are not currently
implemented.

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
- Without appearance cue, only in a separately gated ReID/appearance ablation.
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

- Unique-Entity Recall.
- Duplicate Report Rate.
- False Merge Rate.
- Missed Entity Rate.
- Correct Reacquisition Rate.
- Reconciliation/continuity accuracy.
- Entity Count Error.
- Event precision, recall, and F1.
- False alerts per minute during camera-motion segments.
- Duplicate-alert rate and missed-event rate.
- Evidence completeness and source-to-alert traceability.
- Time-to-alert.
- SAR last-known-location continuity.
- SAR duplicate-victim suppression.

Every metric requires a versioned operational definition, numerator,
denominator, unit, aggregation rule, and handling rule for unavailable data
before final evaluation. Resource feasibility includes end-to-end latency,
throughput, delivered FPS, peak RAM, and peak VRAM under the declared sequential,
batch-one experiment protocol. Compatibility-smoke timings are excluded.

Evidence completeness measures the presence of required reason codes and
evidence references, not confidence alone. Detection confidence and event
confidence are evaluated and stored separately. Event state is evaluated
separately from the later operator-alert state and human actions.

## Data

Use public, identity-annotated UAV sequences. Split data by sequence, flight,
site, or mission so correlated frames from one continuous capture cannot appear
across development and held-out evaluation partitions. Random frame-level
splits are prohibited. Tune thresholds on development sequences and report
final results on held-out sequences. Derived event labels must be versioned
separately from model inputs and must never be used for model training.

Border-surveillance and SAR evaluation must report their mission-specific data,
rules, annotations, and failure modes separately as well as any justified
aggregate. Authorization labels must come from documented mission/access-control
inputs or explicit operator annotations and must not be inferred from
appearance.

Visible-weapon cue evaluation is a separate gated study, not part of the
primary research contribution. It requires suitable held-out public data,
precision/recall measurement, and mission-relevant false-positive and
false-negative analysis before integration. Existing YOLO11n and YOLO26n COCO
checkpoints are not firearm detectors.

## Evidence and safety protocol

Every evaluated reconciliation, event, and candidate alert must retain
machine-readable reason codes and evidence references. Confidence alone is
insufficient. Evidence must preserve the mission/source/frame context,
transient TrackKeys, Persistent Mission Entity hypothesis, applicable rule and
version, uncertainty, and configuration/model provenance needed for replay.

Persistent Mission Entities are operational continuity hypotheses, not
biometric, civil, or confirmed real-world identities. Mission authorization
status uses `authorized`, `unauthorized`, or `unknown`, defaults to `unknown`,
and changes only through an authenticated mission input, documented
access-control input, or explicit authorized-operator action.

Mission authorization status, ally/enemy status, hostility, nationality, and
intent must never be inferred from a face, clothing, ethnicity, religion,
gender, another protected characteristic, or general appearance. Behavior may
contribute to a documented event such as loitering or zone crossing, but it must
not determine nationality, identity, ally/enemy status, or authorization.
`unauthorized` never means enemy.

Mission events and operator alerts have separate states. Candidate alerts
require human review; acknowledgement, dismissal, escalation, and resolution
remain authenticated operator actions. GeoVision AI must never autonomously
recommend, select, or execute force or lethal action. Human review of evidence
does not authorize the system to recommend force.

## Prohibited research claims

- YOLO11n or YOLO26n detects firearms.
- Concealed-weapon detection or visible weapon confirmation without separately
  validated cue evidence.
- Inferring ally/enemy status, authorization, hostility, nationality, or intent
  from protected characteristics or general appearance.
- A Persistent Mission Entity is a confirmed biometric, civil, or real-world
  identity.
- Possible distress constitutes a medical diagnosis.
- Metric distance or location is valid without calibration, reference-frame,
  uncertainty, and provenance evidence.
- Compatibility-smoke timings are publishable benchmark results.
- Deferred or planned capabilities are already implemented.
- Autonomous recommendation, selection, or execution of force or lethal action.

## Milestone 1 perception protocol

The four required detector/tracker results will be generated from two detector
passes over an immutable recorded file. YOLO11n will run once and its normalized
detections will be replayed through ByteTrack and BoT-SORT. YOLO26n will then
follow the same procedure. BoT-SORT uses native `sparseOptFlow` GMC and ReID
remains disabled. Future webcam and RTSP sources are classified as live-smoke
inputs, not comparative matrix sources.

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

At the Milestone 1B checkpoint, `hardware_confirmed` remained false until
YOLO11n and YOLO26n could both pass controlled 640 x 640 FP16 smoke tests on the
target GPU. Historically, both model-weight hashes were unresolved at that
checkpoint, so benchmark execution failed closed. The later Milestone 1C record
below verifies YOLO11n; YOLO26n and final hardware confirmation remain
unresolved.

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

## Milestone 1C YOLO26n checkpoint compatibility evidence

The verified `yolo26n.pt` checkpoint came from the official Ultralytics assets
v8.4.0 release at
`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`.
The controlled download redirected to
`release-assets.githubusercontent.com`. Its canonical SHA-256 is
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`,
and its exact size is 5,544,453 bytes. The checkpoint remains outside Git under
`%LOCALAPPDATA%\GeoVision-AI\models\ultralytics\v8.4.0`.

Ultralytics 8.4.127 loaded the checkpoint through the `YOLO` wrapper as a
`DetectionModel` with architecture `yolo26n.yaml`, scale `n`, detection task,
2,572,280 parameters, and 80 COCO classes. The runtime used Torch
2.13.0+cu130, TorchVision 0.28.0+cu130, and the CUDA 13.0 runtime on the RTX
3050 Laptop GPU with compute capability 8.6. The actual inference device was
`cuda:0`, the model parameter dtype was FP16, and no CPU fallback occurred.

A deterministic in-memory 640 x 640 BGR array was processed at batch size 1
for exactly three warm-up calls and five compatibility-smoke calls. Peak
allocated CUDA memory was 50,759,168 bytes, peak reserved CUDA memory was
69,206,016 bytes, and allocated memory after the final inference was 38,461,440
bytes. No CUDA OOM occurred, no timings were collected, and the package
inventory remained unchanged at 62 packages. All calls produced zero
detections; that synthetic-input result is not an accuracy result.

The Ultralytics `half` argument emitted a deprecation warning even though FP16
remained active. The future detector adapter must address that API deprecation
without silently changing the configured experimental precision. The COCO
checkpoint is not a firearm detector.

This is compatibility evidence, not benchmark evidence. The CUDA memory values
are checkpoint-smoke measurements, not full-pipeline memory requirements.
`hardware_confirmed: true` means only that both configured detector checkpoints
completed bounded 640 x 640, batch-one FP16 CUDA compatibility smoke validation
on the declared RTX 3050 Laptop GPU environment. It does not establish
benchmark performance, dataset accuracy, real-time live-feed performance,
completed detector/tracker adapters, production readiness, weapon detection,
depth or distance validity, SAR capability, full-system memory feasibility, or
simultaneous model residency.
