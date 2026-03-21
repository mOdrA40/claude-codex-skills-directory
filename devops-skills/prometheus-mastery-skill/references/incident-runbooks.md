# Incident Runbooks (Prometheus)

## Cover at Minimum

- Target scrape collapse.
- Rule evaluation backlog.
- Disk or WAL pressure.
- Cardinality explosion.
- Remote write saturation or silent data loss.
- Bad config rollout or broken relabeling.

## Response Rules

- Stabilize data collection on critical targets first.
- Prefer reversible changes before broad config surgery.
- Protect evidence needed for RCA while restoring observability.
- Distinguish product outage from observability outage, then handle overlap carefully.

## Principal Review Lens

- Can on-call restore trustworthy telemetry in 15 minutes?
- Which emergency action risks hiding more truth than it restores?
- What metric or log proves the observability system is actually healthy again?
- Are runbooks specific enough to work during sleep-deprived incidents?
