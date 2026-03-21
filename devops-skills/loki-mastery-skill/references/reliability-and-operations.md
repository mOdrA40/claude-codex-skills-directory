# Reliability and Operations (Loki)

## Operational Defaults

- Monitor ingest health, query latency, index behavior, object storage dependency, and tenant-level load skew.
- Keep upgrades and retention-policy changes staged and reversible.
- Distinguish platform ingest incidents from query-side user pain quickly.
- Document emergency controls for noisy tenants and runaway queries.

## Run-the-System Thinking

- Logging platforms need SLOs if incident response depends on them heavily.
- Capacity planning should include spikes, incident-driven query storms, and compliance retention.
- On-call should know which log classes matter most during outages.
- Simplicity and strong label discipline outperform uncontrolled flexibility.

## Principal Review Lens

- What signal predicts a bad logging day earliest?
- Which tenant or source should be isolated first during stress?
- Can the team explain storage/query architecture clearly enough to operate it?
- Are we running a disciplined log platform or a giant text dump?
