# Incident Runbooks (Flink)

## Cover at Minimum

- Checkpoint failure storm.
- Backpressure meltdown.
- Bad savepoint/upgrade rollout.
- Sink-side failure causing replay pain.
- Event-time correctness incident.
- Multi-tenant resource starvation.

## Response Rules

- Restore correctness and critical pipeline flow before optimization.
- Prefer targeted rollback and throttling over broad platform panic.
- Preserve checkpoint, watermark, and sink evidence for RCA.
- Communicate clearly about correctness, lateness, and replay impact.

## Principal Review Lens

- Can responders reduce blast radius quickly?
- Which emergency action most risks corrupting correctness expectations?
- What proves the platform is healthy again end-to-end?
- Are runbooks realistic for real streaming incidents?
