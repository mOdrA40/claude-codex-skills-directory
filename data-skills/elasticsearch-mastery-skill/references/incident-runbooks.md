# Incident Runbooks (Elasticsearch)

## Rules

- Cover red cluster, yellow cluster, heap pressure, indexing backlog, and slow search incidents.
- Stabilize critical workloads before ideal tuning.
- Include safe and unsafe actions explicitly.
- Recovery should be verified with metrics and user-facing checks.

## Incident Heuristics

### Separate indexing pain from search pain

Some incidents mainly damage freshness and indexing throughput. Others mainly damage user-facing query latency, relevance, or cluster recovery posture. Runbooks should classify that quickly.

### Protect recovery options first

The first actions should reduce heap, shard, or query pressure without making reallocation, mapping rollback, or node recovery harder later.

### Recovery must include search trust

A cluster is not really healthy if shard state improves but users still experience stale results, poor relevance, or unstable latency.

## Common Failure Modes

### Cluster-color fixation

Teams focus on red/yellow/green state while the most important user-facing search or indexing promise remains broken.

### Heap symptom response without workload diagnosis

Immediate pressure is addressed, but the underlying mapping, shard, or query behavior that caused the problem remains unowned.

### Temporary search calm mistaken for recovery

Queries improve briefly, but indexing backlog, GC pressure, or shard movement still guarantee more pain.

## Principal Review Lens

- Can on-call reduce user pain in 10 minutes?
- Which emergency action creates worse recovery later?
- What confirms true recovery instead of temporary relief?
- Which Elasticsearch incident still depends too much on expert intuition rather than an explicit playbook?
