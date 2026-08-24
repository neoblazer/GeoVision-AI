# Implementation Roadmap

This is an ordered roadmap, not a statement that every listed capability
already exists. A capability remains planned until its implementation, tests,
and applicable exit criteria are complete.

Roadmap phase numbers describe execution order; existing M1-M5 identifiers
remain repository milestone names. The current mapping is: Roadmap Phase 2
completes repository M1, Phase 3 extends M2, Phase 5 implements M3, Phase 6
implements M4, and Phase 8 delivers M5.

The clean-foundation checkpoint already provides immutable schemas, ports,
deterministic mocks, a minimal pipeline and API, strict Milestone 1 experiment
contracts, and the verified vision-runtime foundation.

All milestones must remain feasible on Windows with an Intel i5-12500H, 16 GB
RAM, and an RTX 3050 Laptop GPU with 4 GB VRAM. Model execution is sequential,
uses batch size 1, and relies on zero-budget open-source software and pretrained
models. New hardware-dependent capability requires an explicit feasibility
gate.

## Roadmap Phase 1 - Documentation and safety alignment

- Establish the dual `border_surveillance` and `search_and_rescue` mission.
- Record non-lethal, human-in-the-loop boundaries and prohibited claims.
- Require that GeoVision AI must never autonomously recommend, select, or
  execute force or lethal action.
- Define capability status, authorization semantics, evidence responsibility,
  and the primary research question.
- Align architecture, ADRs, delivery order, and research protocol without
  representing planned capabilities as implemented.

Exit criterion: mission, safety, research, and capability-status language is
consistent across the project documentation.

## Roadmap Phase 2 - Complete repository Milestone 1

- Complete YOLO11n and YOLO26n recorded-file detector adapters.
- Complete ByteTrack and BoT-SORT adapters, retaining BoT-SORT
  `sparseOptFlow` GMC and disabled ReID.
- Run each detector once per immutable source and replay normalized cached
  detections through both trackers.
- Implement the deterministic four-combination runner, versioned cache,
  manifests, MOT and machine-readable artifacts, and latency/FPS/RAM/VRAM
  measurements.
- Perform comparative evaluation sequentially with batch size 1.

Exit criterion: one command produces reproducible artifacts for exactly four
detector/tracker combinations from recorded files, with provenance and resource
measurements sufficient to audit every result.

## Roadmap Phase 3 - Camera-motion reliability

This phase extends repository Milestone 2.

- Mask detector boxes before background-motion estimation.
- Compute sparse optical-flow correspondences and a RANSAC transform.
- Record inlier ratio, residual error, transformation, reliability, and
  keyframe-relative stabilized footpoints.
- Evaluate enabled/disabled camera-motion and reliability ablations.

Exit criterion: motion outputs and uncertainty are deterministic, replayable,
and demonstrably improve or safely decline to influence downstream reasoning.

## Roadmap Phase 4 - Mission and evidence contracts

- Add mission context and the two mission modes ahead of perception processing.
- Define source- and tracker-instance-qualified transient TrackKeys.
- Define authorization states and authenticated provenance.
- Add explicit location validity/unavailable semantics.
- Define evidence packets and separate event and operator-alert states.

Exit criterion: strict versioned contracts cover mission-scoped inputs and
evidence without claiming that persistence, alerts, or metric location already
work.

## Roadmap Phase 5 - Persistent Mission Memory

This phase implements repository Milestone 3.

- Implement tracklet lifecycle and mission-scoped Persistent Mission Entities.
- Add hard feasibility gates, reliability-aware cue fusion, one-to-one
  assignment, and immutable association evidence.
- Evaluate continuity, reacquisition, false merges, and entity-count accuracy on
  held-out public sequences and deterministic fragmentation tests.

Exit criterion: reconciliation improves entity-continuity metrics locked before
held-out evaluation without unacceptable false merges or identity claims.

## Roadmap Phase 6 - Deterministic Event Engine

This phase implements repository Milestone 4 for border-surveillance and SAR
events.

- Implement versioned border-surveillance and search-and-rescue rules.
- Keep event confidence separate from detection confidence.
- Require reason codes, evidence references, uncertainty, and temporal
  hysteresis for every event.
- Represent possible distress without medical diagnosis.

Exit criterion: stored inputs reproduce identical reason-coded event records,
and held-out evaluation reports false, missed, and duplicate events.

## Roadmap Phase 7 - Alert evaluation and deduplication

- Deduplicate reports by Persistent Mission Entity rather than transient track.
- Build immutable, versioned evidence packets from event provenance.
- Evaluate duplicate-report suppression, evidence completeness, traceability,
  missed-event rate, false/duplicate alerts, and time-to-alert.

Exit criterion: the proposed system is compared with frame-level and
transient-track reporting using mission metrics locked before evaluated test
results are inspected.

## Roadmap Phase 8 - Operator evidence platform

This phase delivers repository Milestone 5 in two dependent parts.

### Phase 8A - Alert lifecycle and evidence

- Implement candidate-alert creation and human review states independently of
  event state.
- Record authenticated acknowledgement, confirmation, dismissal, escalation,
  resolution, and operator reasons.
- Export versioned evidence packets and audit records over the same stored
  research evidence.

### Phase 8B - Replay and presentation

- Depend on the evidence and alert contracts completed in Phase 8A.
- Provide evidence replay, dashboard inspection, and structured report
  presentation.

Exit criterion: an operator can audit an alert from source evidence through
event reasoning and every human state transition.

## Roadmap Phase 9 - Visible-weapon cue gate

- Select a suitable public dataset and separately validated pretrained cue
  detector under the zero-budget constraint.
- Evaluate visible-weapon precision/recall and mission-relevant false positives
  on held-out sequences.
- Integrate only as operator-flagged risk evidence after passing the gate; never
  as weapon confirmation or an autonomous threat decision.

Quantitative acceptance thresholds, dataset versions and splits, and measurement
procedures remain unresolved. They must be locked before evaluated test results
are inspected. Failure to meet the future gate leaves the capability disabled
or experimental and prevents operational use.

Exit criterion: limitations, failure analysis, provenance, and a human-review
boundary are documented before any mission integration.

## Roadmap Phase 10 - Metric distance/depth gate

- Preserve the `DepthService -> DistanceService -> FusionService` boundary.
- Select a backend only through a separate recorded decision and hardware test.
- Validate calibration, metric scale, uncertainty, provenance, and explicit
  unavailable behavior before reporting distance or location as metric.

Quantitative acceptance thresholds, validation-dataset versions and splits, and
measurement procedures remain unresolved. They must be locked before evaluated
test results are inspected. Failure to meet the future gate leaves the
capability disabled or experimental and prevents operational use.

Exit criterion: held-out validation establishes the supported operating
conditions and every unsupported condition fails unavailable.

## Roadmap Phase 11 - Authorized live-input gate

- Add webcam, CCTV, and authorized RTSP sources only after recorded-file
  reproducibility and reliability are stable.
- Enforce authenticated mission/source registration, bounded buffering,
  reconnect/drop semantics, and evidence provenance.
- Keep live smoke results separate from comparative benchmark evidence.

Quantitative reliability and resource thresholds, versioned evaluation sources
or datasets, and measurement procedures remain unresolved. They must be locked
before evaluated test results are inspected. Failure to meet the future gate
leaves live ingestion disabled or experimental and prevents operational use.

Exit criterion: live sources fail safely, remain resource-feasible, and cannot
enter mission processing without authorization and mission context.

## Roadmap Phase 12 - Later independently gated capabilities

- Evaluate terrain and hazard information with explicit uncertainty.
- Add validated localization and map/reference-frame provenance.
- Consider advanced route planning only as operator decision support.
- Treat adaptive model/frame scheduling as supporting resource optimization.
- Gate multi-drone, thermal/infrared, autonomous-flight, and unavailable
  hardware extensions separately.

Exit criterion: each enabled capability has independent validation, explicit
failure states, hardware feasibility evidence, and a non-lethal human-review
boundary.
