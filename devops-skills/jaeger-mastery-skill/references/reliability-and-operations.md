# Reliability and Operations (Jaeger)

## Operational Defaults

- Monitor collector health, queueing, storage backend latency, query performance, and ingestion skew.
- Keep sampling, backend, and retention changes staged and reversible.
- Distinguish telemetry pipeline loss from backend query degradation quickly.
- Document safe mitigation for overload and storage trouble.

## Run-the-System Thinking

- Tracing backends deserve platform SLOs if incident response depends on them.
- Capacity planning should include incident bursts and backend contention.
- On-call should know what trace workflows matter most when the system is degraded.
- Operational trust depends on honest communication about missing trace truth.

## Principal Review Lens

- What signal predicts platform degradation earliest?
- Which dependency hides the most truth when it fails?
- Can the team explain where traces may be lost or delayed?
- Are we operating a trusted tracing platform or an optimistic archive?
