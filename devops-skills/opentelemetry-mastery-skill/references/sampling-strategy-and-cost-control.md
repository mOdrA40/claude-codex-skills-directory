# Sampling Strategy and Cost Control

## Rules

- Sampling is a product and incident-debugging tradeoff, not only a finance setting.
- Different workloads may justify different sampling postures.
- Head sampling, tail sampling, and error-focused sampling each have operational consequences.
- Cost controls should not erase the very traces needed during major incidents.

## Practical Guidance

- Preserve critical paths, high-latency outliers, and error traces deliberately.
- Align collector capacity and backend limits with sampling goals.
- Review how sampling affects correlation with logs and metrics.
- Make sampling policy visible to service owners.

## Principal Review Lens

- Which traces become unavailable exactly when they matter most?
- Is sampling bias distorting architecture decisions?
- What workload should be sampled differently based on business criticality?
- Are we saving money intelligently or just blinding ourselves cheaply?
