# Schema Evolution (MongoDB)

## Rules

- Flexible schema does not eliminate migration discipline.
- Version documents intentionally when shapes evolve.
- Readers and writers may coexist across schema versions for a while.
- Backfills should be observable, throttled, and reversible where possible.

## Evolution Heuristics

### Flexibility is not operational forgiveness

MongoDB allows mixed document shapes, but that does not remove the need for clear compatibility windows, ownership, and migration choreography.

### Mixed-version reality must be designed

Applications should explicitly handle:

- old reads on new writers
- new reads on old documents
- partial backfill windows
- delayed cleanup of deprecated fields or shapes

### Backfills are product-risk events

A large document migration can change write pressure, query behavior, and failure handling in ways that are easy to underestimate if the team focuses only on code compatibility.

## Common Failure Modes

### Flexibility theater

The team treats schema evolution casually because the database permits it, but application logic, analytics, or downstream contracts are more rigid than admitted.

### Infinite compatibility window

Old and new document shapes coexist longer than intended until the system effectively supports both forever.

### Migration without consumer mapping

Documents evolve, but no one has a clear map of which services, jobs, or dashboards still depend on the old shape.

## Principal Review Lens

- How many schema versions can exist safely at once?
- What breaks first if backfill lags for days?
- Is the application resilient to mixed-shape documents?
- What compatibility promise is currently implied but not explicitly owned?
