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

## Principal Review Lens

- Which warehouse is paying the highest noisy-neighbor tax?
- Are we optimizing for user experience or just warehouse uptime?
- What workload should be isolated first?
- Can the team explain warehouse cost behavior clearly?
