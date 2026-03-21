# Cluster Topology and Operations

## Rules

- Topology should reflect failure domains, latency goals, and operational maturity.
- Superclusters, leaf nodes, and gateways solve different organizational problems.
- Simpler topologies are easier to reason about under stress.
- Plan capacity and failover based on connection, stream, and traffic realities.

## Operational Guidance

- Distinguish local autonomy needs from global routing needs.
- Observe cluster membership, route health, stream replication, and client reconnect behavior.
- Test partition, node loss, and reconnect scenarios with real clients.
- Keep operator responsibilities for topology changes explicit.

## Principal Review Lens

- What topology feature are we using that we cannot confidently operate?
- Which failure domain still has surprising blast radius?
- Are clients robust to reconnect and reroute behavior?
- What topology simplification would improve reliability most?
