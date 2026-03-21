# Locality and Zone Configs (CockroachDB)

## Rules

- Locality settings should express latency and survivability intent.
- Keep placement logic understandable by humans on-call.
- Avoid accidental cross-region tax on hot paths.
- Validate configuration against actual client geography.

## Placement Heuristics

### Model locality as a business decision

Placement is not only a database configuration detail. It defines which users pay latency, which data survives which failures, and how operable the cluster remains during regional stress.

### Prefer explainable placement over clever placement

If operators cannot quickly explain where data, replicas, and leaseholders should live for a critical workload, the design is too opaque for production safety.

### Test degraded-region behavior, not only happy-path geography

A placement model that looks correct during normal traffic may fail badly when one region is impaired, slow, or partially partitioned.

## Common Failure Modes

### Geography theater

The cluster is multi-region by topology, but the dominant workload still pays unnecessary cross-region latency because access paths and placement were never aligned.

### Survivability without operability

The system technically meets failure goals, but operator understanding of placement is too weak to troubleshoot incidents confidently.

### Configuration drift without workload review

Zone and locality choices remain unchanged while product geography, tenant distribution, or access patterns evolve underneath them.

## Principal Review Lens

- Which requests cross regions unnecessarily?
- Does placement match business criticality and data sovereignty needs?
- Can operators predict where data and leaseholders live?
- Which locality rule will become hardest to reason about during a real regional incident?
