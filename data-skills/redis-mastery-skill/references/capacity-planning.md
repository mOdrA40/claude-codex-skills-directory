# Capacity Planning (Redis)

## Rules

- Capacity is mostly about memory, hot traffic, and persistence cost.
- Plan headroom for failover, fragmentation, and warm-up.
- Growth models should include key cardinality explosion.
- Benchmark with realistic value sizes and access distributions.

## Principal Review Lens

- What breaks first under 2x traffic: memory, latency, or eviction?
- How much safe headroom exists during failover?
- Are growth forecasts based on real key distributions?
