# Schema Evolution (MongoDB)

## Rules

- Flexible schema does not eliminate migration discipline.
- Version documents intentionally when shapes evolve.
- Readers and writers may coexist across schema versions for a while.
- Backfills should be observable, throttled, and reversible where possible.

## Principal Review Lens

- How many schema versions can exist safely at once?
- What breaks first if backfill lags for days?
- Is the application resilient to mixed-shape documents?
