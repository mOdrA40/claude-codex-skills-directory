# Capacity Planning (Kafka)

## Rules

- Capacity planning must include retention, replication, and replay spikes.
- Partition count affects throughput, operations, and recovery time.
- Benchmarks should use real message sizes and key distributions.
- Storage and network are often the real budget constraints.

## Principal Review Lens

- What fails first under 2x event volume?
- Is partition count helping throughput or only adding ops tax?
- How much headroom exists during broker loss or replay storm?
