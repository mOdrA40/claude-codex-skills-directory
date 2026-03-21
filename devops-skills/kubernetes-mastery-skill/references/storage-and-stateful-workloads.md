# Storage and Stateful Workloads

## Rules

- Stateful workloads need durability, failover, and restore thinking upfront.
- PVCs and storage classes should match workload IOPS and failure needs.
- StatefulSet use should be justified, not habitual.
- Data plane behavior under reschedule matters.

## Principal Review Lens

- What happens to data on node loss right now?
- Is this stateful workload truly safe to run on the cluster?
- Which storage assumption breaks first under pressure?
