# Contention and Hotspots (CockroachDB)

## Rules

- Global counters, central allocators, and single hot rows are red flags.
- Contention must be designed away before scaling it away.
- Watch retry storms as a symptom of write concentration.
- Hotspot remediation often needs data model changes, not tuning only.

## Principal Review Lens

- Which key or row is absorbing disproportionate writes?
- How does contention change by region and workload mix?
- Are retries masking a data-model problem?
