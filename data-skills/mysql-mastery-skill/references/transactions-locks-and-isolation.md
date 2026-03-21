# Transactions, Locks, and Isolation

## Rules

- Transaction boundaries should protect real invariants and stay short.
- Lock behavior must be understood under concurrency, not only in isolated tests.
- Isolation choices have correctness and throughput tradeoffs.
- Long-running transactions create invisible operational cost.

## Design Guidance

- Identify where row, gap, or metadata locks may appear.
- Never hold DB transactions open across slow network calls casually.
- Build retry and idempotency around known contention or deadlock paths.
- Make correctness needs explicit before choosing stronger isolation.

## Principal Review Lens

- Which invariant truly requires transactional coupling here?
- What concurrency pattern creates the worst lock shape?
- Are we blaming MySQL for application transaction design mistakes?
- What deadlock or lock wait will become common under growth?
