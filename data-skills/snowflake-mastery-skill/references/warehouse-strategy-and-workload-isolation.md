# Warehouse Strategy and Workload Isolation

## Rules

- Warehouses should be designed around workload classes, concurrency, latency, and cost behavior.
- Shared warehouses create convenience and noisy-neighbor risk together.
- Isolation decisions should reflect business criticality and operator needs.
- Performance tuning must include cost tradeoffs explicitly.

## Practical Guidance

- Separate critical BI, ELT, data science, and ad-hoc exploration where justified.
- Use scaling and suspend/resume settings based on real usage patterns.
- Track warehouse usage by team and purpose.
- Keep platform defaults simple enough for predictable behavior.

## Isolation Heuristics

### Separate by business risk, not just team names

Workload isolation should reflect:

- user-facing latency sensitivity
- cost unpredictability
- concurrency spikes
- impact of one workload on another

### Shared warehouses need explicit tradeoff ownership

Shared infrastructure is not free convenience. Someone should own the noisy-neighbor risk, cost attribution, and escalation path.

## Common Failure Modes

### Cheap-looking defaults, expensive reality

Warehouses start simple and shared, then gradually absorb incompatible workloads until cost and user experience become hard to explain.

### Isolation too late

Teams wait until incidents or finance pressure become serious before separating critical workloads from exploratory ones.

## Principal Review Lens

- Which warehouse is paying the highest noisy-neighbor tax?
- Are we optimizing for user experience or just warehouse uptime?
- What workload should be isolated first?
- Can the team explain warehouse cost behavior clearly?
- Which warehouse decision looks operationally simple today but creates finance or latency pain later?
