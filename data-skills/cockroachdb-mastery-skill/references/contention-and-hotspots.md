# Contention and Hotspots (CockroachDB)

## Rules

- Global counters, central allocators, and single hot rows are red flags.
- Contention must be designed away before scaling it away.
- Watch retry storms as a symptom of write concentration.
- Hotspot remediation often needs data model changes, not tuning only.

## Contention Model

In CockroachDB, contention is rarely just a throughput issue. It is usually a data-model and access-pattern issue that eventually appears as latency spikes, restart errors, and unstable tail behavior.

High-risk patterns include:

- one row or narrow key range absorbing many writes
- leaseholder placement mismatched with dominant writers
- globally ordered sequences or allocators in high-write paths
- transactional fan-in where many requests compete on the same business entity

## Practical Heuristics

### Design away central write points

If many concurrent requests must update one logical counter, allocator, or row, the design is already on a dangerous path.

### Treat retries as a clue, not a solution

Application retries may hide contention temporarily while increasing total work and making tail latency worse.

### Measure by workload slice

Look at contention by:

- key pattern
- tenant or workload class
- region
- transaction type

Cluster averages hide hotspots badly.

## Common Failure Modes

### Retry storm camouflage

The system appears resilient because requests eventually succeed, but user latency and infrastructure waste are both climbing.

### Regional hotspot amplification

Multi-region topology spreads the cluster, but one write-heavy key range still dominates latency because placement and write pattern do not match.

### Tuning without model change

Teams try to solve a schema and access-pattern problem with infrastructure scaling or generic tuning only.

## Principal Review Lens

- Which key or row is absorbing disproportionate writes?
- How does contention change by region and workload mix?
- Are retries masking a data-model problem?
- What design change would remove the hotspot rather than merely delaying it?
