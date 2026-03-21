# Data Modeling and Shard-Aware Partitioning (ScyllaDB)

## Rules

- Model data for query paths, partition balance, and shard-local efficiency.
- Partition keys still define scale behavior and hotspot risk.
- Shard-aware client and workload assumptions should be validated, not presumed.
- Avoid designs that create oversized partitions or cross-shard pain.

## Design Guidance

- Validate partition cardinality and hottest-key distribution explicitly.
- Align table design with expected read/write concurrency.
- Duplicate data where it simplifies stable high-throughput access.
- Keep tenant and time dimensions visible in load planning.

## Modeling Heuristics

### Low latency depends on stable partition behavior

ScyllaDB rewards models that keep partition access predictable, shard-local assumptions valid, and hottest paths boring even under very high concurrency.

### Shard awareness should improve confidence, not replace measurement

The presence of shard-aware clients or design ideas is not enough unless the team validates that real workloads actually benefit and remain balanced.

### Optimize for worst-case hotspots

The model should be judged by the heaviest tenant, burstiest event class, and most concurrency-sensitive path—not by median traffic alone.

## Common Failure Modes

### Shard-awareness theater

Teams assume ScyllaDB-specific performance benefits will arrive automatically while partitioning and workload shape still create the dominant pain.

### Median-latency comfort

Average latency looks excellent, but hotspot partitions or tenant-local incidents violate the low-latency promise where it matters most.

### Growth-path blindness

The design works well early, then falls apart when key concentration or concurrency rises beyond what the chosen partition model can distribute safely.

## Principal Review Lens

- Which partition pattern breaks low-latency assumptions first?
- Are we relying on shard-aware behavior we have not measured?
- What key choice becomes dangerous at 10x traffic?
- Which table should be redesigned before scale magnifies it?
- Which partition or tenant pattern is most likely to create a deceptive “healthy average, sick hotspot” situation?
