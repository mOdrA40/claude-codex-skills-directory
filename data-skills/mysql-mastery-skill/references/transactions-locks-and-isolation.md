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

## Isolation Heuristics

### Choose isolation for the invariant, not the team habit

Stronger isolation can be necessary, but it always costs something in throughput, lock behavior, or operator simplicity.

### Understand lock shape under realistic concurrency

It is not enough to know the query is correct. Teams should know what kind of lock footprint it creates under competing writers and readers.

## Common Failure Modes

### Deadlock surprise as growth arrives

The application appears stable in early scale, but one growth step or one new worker class reveals lock-order problems that were already latent.

### Isolation by cargo cult

Teams choose stronger semantics because they sound safer without proving the business invariant needs them.

### Retry hides transaction design debt

Deadlocks and lock waits become "expected" without redesigning the flows that create them.

## Principal Review Lens

- Which invariant truly requires transactional coupling here?
- What concurrency pattern creates the worst lock shape?
- Are we blaming MySQL for application transaction design mistakes?
- What deadlock or lock wait will become common under growth?
- Which transaction path should be simplified before throughput growth turns it into an operator problem?
