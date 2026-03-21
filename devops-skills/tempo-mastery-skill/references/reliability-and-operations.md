# Reliability and Operations (Tempo)

## Operational Defaults

- Monitor ingest health, compaction, object storage dependency, search performance, and tenant load skew.
- Keep backend, config, and retention changes staged and reversible.
- Distinguish collector-side telemetry loss from Tempo-side platform degradation.
- Document safe mitigation paths for ingest surges and storage trouble.

## Run-the-System Thinking

- Tempo deserves SLOs if production debugging depends on it heavily.
- Capacity planning should include incident bursts, not only baseline traffic.
- On-call should know which trace workflows matter most during outages.
- Operational trust depends on clear communication of trace availability and limits.

## Principal Review Lens

- What signal predicts trace-platform degradation earliest?
- Which dependency failure hides the most truth?
- Can the team explain where traces might be lost in the pipeline?
- Are we operating a trusted backend or a hopeful archive?
