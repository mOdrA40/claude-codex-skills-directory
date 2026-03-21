# Incident Runbooks (Flink)

## Cover at Minimum

- Checkpoint failure storm.
- Backpressure meltdown.
- Bad savepoint/upgrade rollout.
- Sink-side failure causing replay pain.
- Event-time correctness incident.
- Multi-tenant resource starvation.

## Incident Heuristics

### Separate flow failure from correctness failure

Some Flink incidents stop throughput. Others keep the job running while corrupting lateness expectations, replay behavior, or sink-side correctness.

### Protect state safety during recovery

Rollback, restart, or upgrade actions should be judged not only by speed but by how safely they preserve state, checkpoint integrity, and consumer trust.

### Recovery must include downstream semantics

A stream is not healthy again if Flink is processing but sinks, consumers, or downstream contracts still experience duplicates, lateness, or inconsistent revisions.

## Response Rules

- Restore correctness and critical pipeline flow before optimization.
- Prefer targeted rollback and throttling over broad platform panic.
- Preserve checkpoint, watermark, and sink evidence for RCA.
- Communicate clearly about correctness, lateness, and replay impact.

## Common Failure Modes

### Throughput recovery without semantic recovery

The pipeline moves again, but event-time behavior, sink duplicates, or late-data revision expectations remain broken.

### Savepoint confidence without rollback realism

Teams trust the mechanics of savepoints or upgrades more than their actual tested recovery path under incident pressure.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks corrupting correctness expectations?
- What proves the platform is healthy again end-to-end?
- Are runbooks realistic for real streaming incidents?
- Which Flink incident class still lacks a clearly safe degraded-mode response?
