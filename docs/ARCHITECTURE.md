# Architecture

## Design rules

1. Mission context is established and validated before perception processing.
2. Domain schemas do not import FastAPI, OpenCV, Ultralytics, database clients,
   or frontend code.
3. Model implementations live behind ports in `adapters/`.
4. Raw tracker IDs and persistent mission entity IDs are different concepts.
5. Every frame carries an immutable source ID, frame number, and timestamp.
6. Missing or unvalidated metric distance is represented explicitly; it is never
   silently substituted with relative depth.
7. Every entity reconciliation and mission alert records machine-readable
   evidence.
8. Event state and operator-alert state are separate lifecycles.
9. Detection confidence and event confidence are separate values; confidence
   alone is never a substitute for reason codes and evidence references.
10. Experimental code lives outside production packages and is never vendored.

## Target data flow

```text
Authenticated mission context
  -> source access authorization
  -> input service
  -> frame packet
  -> detector adapter
  -> tracker adapter
  -> camera-motion service
  -> reliability-aware reconciliation
  -> persistent Mission Memory
  -> deterministic Event Engine
  -> API / replay / analytics / structured report
```

Mission context includes the mission identifier, mission mode, authorized
sources, applicable zones, and authenticated or operator-supplied mission
inputs. A frame must not enter mission processing without being bound to this
context. Support for a capability in the target flow does not mean that the
capability is currently implemented.

Depth, distance, semantic segmentation, and scheduling are parallel gated
services. Their failures must not stop the core detection/tracking pipeline.
The `DistanceService` abstraction remains the architectural boundary; this
document does not select or claim a depth backend.

## Stable identifiers

- `source_id`: one input stream or recorded video.
- `frame_id`: monotonically increasing within one source.
- `track_id`: transient integer emitted by one tracker instance.
- `TrackKey`: transient, source-qualified and tracker-instance-qualified key
  composed eventually from `source_id`, `tracker_instance_id`, and `track_id`.
- `entity_id`: mission-scoped persistent identifier owned by Mission Memory.
- `event_id`: immutable event record with rule and evidence provenance.

A bare tracker ID is not globally unique and must never become a persistent
identity by convention. Until a source-qualified and tracker-instance-qualified
`TrackKey` contract is implemented, downstream code must treat the existing
`track_id` as local to its tracker run and source.

A Persistent Mission Entity links mission observations under an explicit,
evidence-backed reconciliation hypothesis. It is an operational hypothesis for
continuity within a mission, not a biometric identity, civil identity, or proof
that observations depict the same real-world person. Reconciliation must retain
its cues, reason codes, uncertainty, alternatives where applicable, and source
provenance.

## Authorization taxonomy

### Source access authorization

Source access authorization is permission to ingest a webcam, CCTV, UAV, or
RTSP source. It concerns stream ownership, access, and privacy. It does not
describe the authorization, identity, nationality, or intent of any observed
person or vehicle.

### Mission authorization status

A future mission-authorization input uses exactly these states:

- `authorized`
- `unauthorized`
- `unknown`

The default is `unknown`. Mission authorization status applies to an operational
entity within a documented mission or zone. It may change only through an
authenticated mission input, documented access-control evidence, or an explicit
action by an authorized operator.

Mission authorization status, ally/enemy status, hostility, nationality, and
intent must never be inferred from a face, clothing, ethnicity, religion,
gender, another protected characteristic, or general appearance. Behavior may
contribute to a documented event such as loitering or zone crossing, but it must
not determine nationality, identity, ally/enemy status, or authorization.

`unauthorized` means that permission for the relevant mission or zone is not
established. It does not mean enemy, hostile, or a permission to use force. The
authorization value, applicable scope, source, effective time, and provenance
must be retained so later decisions are auditable.

## Evidence and operator-alert lifecycle

The planned Event Engine produces deterministic, reason-coded event records. An
event can assemble an immutable evidence packet containing the relevant frames,
detections, transient TrackKeys, Persistent Mission Entity references, temporal
context, applicable mission or zone rules, confidence components, uncertainty,
and configuration/model provenance.

An event and its evidence may create a candidate operator alert. Alert state is
managed separately from event state through a lifecycle such as pending human
review, acknowledged, dismissed, escalated, or resolved. State transitions must
record the responsible authenticated actor, timestamp, reason, and evidence
version. Changing alert state does not rewrite the underlying event or evidence.

All mission alerts require human review. The operator remains responsible for
interpretation and action, including assessment of an operator-flagged risk
evidence item or possible distress. GeoVision AI must never autonomously
recommend, select, or execute force or lethal action. Human review of evidence
does not authorize the system to recommend force.

## Distance and location validity

Distance and location outputs must expose uncertainty, calibration state,
coordinate/reference frame, method, timestamp, source observations, model or
sensor provenance, and an explicit validity or unavailable state. A numeric
value must not be emitted as metric when calibration, scale, geometry, or
localization validity cannot be established. In that case the service fails
closed with `unavailable` and a machine-readable reason rather than substituting
relative depth or an assumed location.

Last-known-location records inherit these requirements. They must distinguish
an observed or validly estimated location from an image-space point, stale
location, or unavailable location and retain the time and evidence from which
the result was derived.

## Milestone 1 configuration boundary

Versioned YAML defines benchmark semantics and is validated into immutable
domain contracts. The loader does not consult application `Settings` or the
environment. Tracker profile files are resolved by the configuration layer and
validated before a four-way detector/tracker matrix is exposed.

Milestone 1A adds no model or source implementation. Future Ultralytics tracker
internals remain isolated behind the existing tracker port. The standalone
camera-motion service stays downstream and outside Milestone 1; BoT-SORT's
native `sparseOptFlow` GMC is internal to its approved tracker profile.

A `PipelineResult` is one frame-scoped unit. Every contained detection and track
must carry exactly the same immutable `FrameRef` as the enclosing result so
cache, replay, and later evaluation cannot silently mix frames.
