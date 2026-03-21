# Online Migrations and Operational Change

## Rules

- Schema changes on live systems should be staged with lock and rewrite awareness.
- Expand-and-contract patterns are usually safer than one-shot changes.
- Backfills need throttling, visibility, and rollback posture.
- Application and schema rollout order must be coordinated.

## Failure Modes

- Blocking DDL on hot tables without a fallback plan.
- Mixed-version applications incompatible with transitional schema.
- Backfills saturating replicas or overwhelming primary write paths.
- Successful migration commands that still create operational instability.

## Migration Heuristics

### Online change is still production load

Even when tooling reduces lock risk, schema change and backfill work still compete with real traffic, replicas, and operational headroom.

### Expand-and-contract needs explicit observation points

The safest migrations usually create deliberate pause boundaries between schema introduction, app rollout, data backfill, and cleanup so teams can observe real effects before continuing.

### Compatibility windows must be intentional

If old and new application versions may coexist, the migration plan should state clearly what each version can read, write, and safely ignore.

## Additional Failure Modes

### Online-tool overconfidence

Teams trust tooling names like online or non-blocking more than the actual workload, replica, and backfill realities they are about to stress.

### Partial success ambiguity

The change technically progressed, but the system is left in a mixed state that operators and application teams do not fully understand.

### Cleanup rushed before stability is proven

The team removes compatibility layers or old columns too early, turning a recoverable rollout into a higher-risk incident.

## Principal Review Lens

- What is the lock risk and data-copy risk of this change?
- Can old and new app versions coexist safely?
- Which backfill step is most dangerous at peak traffic?
- What recovery plan exists if the migration halfway succeeds?
- Which “online” assumption in this rollout is least trustworthy under real production load?
