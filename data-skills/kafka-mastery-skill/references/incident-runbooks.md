# Incident Runbooks (Kafka)

## Rules

- Cover lag storms, broker loss, rebalance thrash, disk pressure, and poison events.
- Stabilize user-visible backlog and critical consumers first.
- Include safe remediation and explicit anti-actions.
- Recovery must be tied to measurable lag and throughput signals.

## Incident Heuristics

### Classify by failure surface first

Operators should quickly determine whether the main issue is:

- broker or storage pressure
- consumer lag and rebalance instability
- poison-message / replay pain
- producer-side publish degradation
- one topic or tenant dominating shared cluster health

### Protect critical flows before broad cleanup

The right first move is usually to isolate the highest-business-value topics or consumer groups rather than chase perfect global cluster health immediately.

### Recovery must include semantic safety

Kafka is not truly recovered if offsets move again but replay risk, duplicate side effects, or backlog truth remain unclear.

## Common Failure Modes

### Lag improvement mistaken for business recovery

Metrics look better, but consumer correctness, replay pressure, or downstream load is still unsafe.

### Generic cluster response to one-topic pathology

The team treats a localized partition, topic, or consumer problem like a full-cluster failure and creates broader disruption than necessary.

### Rebalance calm without throughput truth

A noisy rebalance wave ends, but the real throughput bottleneck or hot path remains unresolved.

## Principal Review Lens

- Can on-call isolate the failing topic or consumer quickly?
- Which action risks data loss or wider replay pain?
- What proves recovery versus temporary backlog movement?
- Which Kafka incident still lacks a low-regret first-response path?
