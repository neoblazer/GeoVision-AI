# GeoVision AI

GeoVision AI is a research-first UAV video intelligence platform for persistent,
explainable mission reasoning. It converts transient detector/tracker output into
persistent mission entities and auditable surveillance or search-and-rescue events.

## Research question

Can a lightweight, camera-motion-aware Mission Memory reduce duplicate entity
reports and false mission alerts under UAV camera motion and track fragmentation,
without making a consumer laptop pipeline unusably slow?

## Locked constraints

- Free and open-source software only.
- Pretrained models; no custom-dataset-scale training or fine-tuning.
- Windows laptop: Intel i5-12500H, 16 GB RAM, RTX 3050 Laptop GPU (4 GB).
- Inputs: webcam, recorded video, RTSP/CCTV stream, and uploaded video.
- Depth, segmentation, dashboard, replay, scheduling, and LLM summaries are
  supporting platform features, not the paper's headline contribution.

## Current milestone

Milestone 0 establishes stable schemas and interfaces before installing heavy
vision dependencies. The repository currently includes:

- FastAPI application factory and health/config routes.
- Typed detection, tracking, motion, distance, mission-entity, association, and
  event schemas.
- Detector and tracker protocols.
- Deterministic mock detector/tracker implementations.
- A minimal perception pipeline and standard-library unit tests.
- Research protocol and architectural decision records.

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

## Local setup

From PowerShell on Windows:

```powershell
cd geovision-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".\backend[dev]"
python -m unittest discover -s backend\tests -v
uvicorn geovision.main:app --app-dir backend\src --reload
```

Then open `http://127.0.0.1:8000/api/v1/health`.

Heavy model packages and weights are deliberately not installed or committed in
Milestone 0. They will be introduced through isolated adapters after the baseline
tests pass.

