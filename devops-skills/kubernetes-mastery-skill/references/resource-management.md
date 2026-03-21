# Resource Management

## Rules

- Requests and limits should reflect measured workload behavior.
- Mis-set limits create throttling, eviction, and false autoscaling signals.
- Resource policy should balance fairness and reliability.
- Watch for noisy-neighbor patterns across namespaces.

## Principal Review Lens

- Are requests truthful enough for scheduling decisions?
- Which limit is causing latency or restart pain?
- What workload can starve others today?
