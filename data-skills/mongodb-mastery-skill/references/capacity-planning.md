# Capacity Planning (MongoDB)

## Rules

- Plan for working set, storage growth, index growth, and replication overhead.
- Memory pressure and page churn often warn before total failure.
- Capacity models must include shard imbalance scenarios.
- Growth planning should include backup and restore windows.

## Principal Review Lens

- What resource fails first under 2x traffic?
- Are indexes or documents driving growth faster?
- How much headroom exists during elections or maintenance?
