# Capacity Planning (CockroachDB)

## Rules

- Plan for replicas, regions, storage growth, and rebalance cost together.
- Capacity should consider retry amplification and failure headroom.
- Watch skew: hot ranges make averages lie.
- Growth forecasts must include index and replica multiplication.

## Principal Review Lens

- What fails first under 2x traffic or 2x data?
- Is capacity dominated by storage, latency, or contention?
- How much headroom exists during rebalance or node loss?
