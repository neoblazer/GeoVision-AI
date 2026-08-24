# GeoVision AI

GeoVision AI is a non-lethal, human-in-the-loop decision-support and
evidence-reporting platform for resource-constrained UAV operations. It is
intended to help authorized operators review observations from border-area
surveillance and search-and-rescue missions; it does not identify enemies or
diagnose people.

The two mission modes are:

- `border_surveillance`
- `search_and_rescue`

The primary research novelty is **Persistent Mission-Oriented Intelligence
with evidence-backed event reasoning for resource-constrained UAV border
surveillance and search-and-rescue missions.** Persistent Mission Entities,
Mission Memory, and a deterministic Event Engine are the primary research
contribution. Detector and tracker integration is the engineering platform that
supports that research.

## Human-in-the-loop boundary

GeoVision may eventually assemble reason-coded evidence for an operator's
review. Mission events remain machine-generated observations, while alert
acknowledgement, interpretation, escalation, and operational decisions remain
the responsibility of an authorized human. GeoVision AI must never
autonomously recommend, select, or execute force or lethal action. Human review
of evidence does not authorize the system to recommend force.

## Capability status

### Implemented foundation

- Immutable domain schemas, detector/tracker ports, deterministic mocks, and a
  minimal perception pipeline.
- Versioned Milestone 1 experiment contracts with strict, deterministic
  configuration validation.
- A verified CUDA vision runtime on the target Windows laptop.
- Direct ByteTrack and BoT-SORT API smoke validation using externally supplied
  synthetic detections; BoT-SORT used `sparseOptFlow` GMC with ReID disabled.
- Recorded provenance and a controlled compatibility smoke for the YOLO11n
  checkpoint. This is not an accuracy or publishable performance result.

### Immediate Milestone 1 work

- Recorded-file detector and tracker adapters for the four approved
  YOLO11n/YOLO26n and ByteTrack/BoT-SORT combinations.
- One detector pass per detector and source, normalized detection caching, and
  replay of the same detections through both trackers.
- A deterministic runner, run manifests, MOT and machine-readable artifacts,
  resource measurements, and comparative evaluation.

These adapters, runner, caches, and comparative results are not yet
implemented merely because their contracts or dependencies exist.

### Planned mission capabilities

- Persistent Mission Memory and mission-scoped entity reconciliation.
- An evidence-backed deterministic Event Engine with border-surveillance and
  search-and-rescue rules.
- Entity-based report deduplication, command-centre alerts, replay, and a
  dashboard for human review.
- Authorized live webcam, CCTV, and RTSP sources.
- Separately validated visible-weapon cues, metric distance, terrain and hazard
  information, and localization.

These are intended final-product capabilities, but each remains gated on its
own implementation, evidence, and validation. Existing COCO detector
checkpoints must not be represented as firearm detectors.

### Research extensions or separately gated capabilities

- ReID ablations after identity benefit and compute cost are measured.
- Advanced route planning and adaptive model or frame scheduling.
- Multi-drone operation, thermal or infrared sensing, autonomous flight, and
  other hardware not currently available to the project.

### Prohibited capabilities and claims

- Inferring ally/enemy status, mission authorization, hostility, nationality,
  or intent from a face, clothing, ethnicity, religion, gender, another
  protected characteristic, or general appearance.
- Using behavior to determine nationality, identity, ally/enemy status, or
  authorization. Behavior may contribute only to a documented event such as
  loitering or zone crossing.
- Setting mission authorization from anything other than authenticated or
  documented mission inputs or explicit authorized-operator action.
- Autonomous hostile classification.
- Concealed-weapon claims or unvalidated weapon confirmation.
- Biometric identification, civil-identity claims, or treating a Persistent
  Mission Entity as a confirmed real-world identity.
- Medical diagnosis or presenting possible distress as a diagnosis.
- Unvalidated metric distance or location claims.

## Project constraints

- Windows and Python 3.11.
- Intel i5-12500H, 16 GB RAM, and RTX 3050 Laptop GPU with 4 GB VRAM.
- Zero budget, open-source software, and pretrained models only.
- Sequential execution with batch size 1 to remain feasible on the available
  hardware.
- Reproducible, versioned evidence suitable for a publishable research paper.

## Repository layout

```text
backend/
  src/geovision/
    adapters/       # External model/tool implementations
    api/            # HTTP/WebSocket transport
    core/           # Configuration and cross-cutting concerns
    domain/         # Stable domain schemas
    ports/          # Interfaces implemented by adapters
    services/       # Application orchestration
  tests/
configs/            # Versioned experiment configurations
docs/               # Architecture and research protocol
scripts/            # Repository checks and setup helpers
```

## Current engineering checkpoint

The clean foundation and Milestone 1 protocol/configuration work are present,
and the pinned vision runtime dependencies have been introduced and validated.
YOLO11n checkpoint provenance and GPU compatibility are recorded. YOLO26n
checkpoint validation and the real detector/tracker adapter implementation are
still outstanding, so benchmark execution continues to fail closed.

## Local setup

From PowerShell on Windows:

```powershell
cd geovision-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".\backend[dev]"
python -m pytest
uvicorn geovision.main:app --app-dir backend\src --reload
```

Then open `http://127.0.0.1:8000/api/v1/health`.

Model weights, datasets, generated results, and local environments remain
outside Git. Planned capabilities must not be described as implemented until
their corresponding milestone exit criteria and validation evidence are met.
