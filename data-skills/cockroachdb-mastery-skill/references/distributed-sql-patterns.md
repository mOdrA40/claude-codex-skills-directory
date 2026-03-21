# Distributed SQL Patterns (CockroachDB)

## Defaults

- Design with locality and contention in mind.
- Prefer keys and access patterns that avoid global hotspots.
- Keep transactions short and retry-safe.
- Choose regional placement based on latency and survivability goals.

## Common Pitfalls

- Global counters and central allocators.
- Chatty multi-step write flows crossing regions.
- Ignoring transaction retries in application code.

## Principal Review Lens

- Which rows become global contention points?
- What is the cross-region latency tax for this transaction?
- Can this workload tolerate retry semantics cleanly?
