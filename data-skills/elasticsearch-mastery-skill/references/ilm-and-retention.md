# ILM and Retention (Elasticsearch)

## Rules

- Retention is a product and compliance decision, not only a storage setting.
- ILM should align with data freshness, cost, and restore expectations.
- Hot/warm/cold tiering must justify its operational complexity.
- Deletion and rollover behavior should be predictable under load.

## Lifecycle Heuristics

### ILM should encode why data exists

A good lifecycle policy reflects the real value of data over time: hot for active use, warm for lower-latency history, cold for compliance or rare investigation, and deleted when value no longer justifies cost or risk.

### Tiering must earn its complexity

Hot/warm/cold design is useful only when it improves economics and recovery posture enough to justify the extra operational reasoning it requires.

### Rollover and deletion are reliability events too

Lifecycle automation should be understood not just in happy-path storage terms, but in how it behaves during cluster pressure, restore needs, and incident investigation.

## Common Failure Modes

### Retention by fear

Teams keep data indefinitely because deletion feels risky, while cost, search quality, and cluster recovery all worsen.

### Tiering theater

The cluster uses hot/warm/cold language, but the policy is weakly justified and operators cannot clearly explain the real economic or recovery benefit.

### ILM surprise during incidents

Lifecycle transitions continue behaving automatically while responders lack a clear understanding of what that means for recovery, freshness, or retained evidence.

## Principal Review Lens

- Is retention longer than its real value?
- What tier transition creates most operational risk?
- Can the team explain ILM behavior during an incident?
- Which lifecycle rule is least trustworthy under cluster stress or restore pressure?
