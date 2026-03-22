# Service Boundaries and Domain Design on the BEAM

## Principle

The BEAM makes concurrency natural, but concurrency is not architecture. Domain boundaries, message design, and ownership still need to be explicit.

## Boundary Direction

A healthy BEAM service usually keeps a clear direction:

`transport -> orchestration/use-case -> domain -> persistence/adapters`

Phoenix, Broadway, channels, and consumers should enter the system at the edge. They should not become the domain model.

## Rules

- keep domain boundaries coherent before choosing process topology
- do not replace design with more processes
- decide which responsibilities deserve long-lived process state and which do not
- keep transport concerns at the edge even in Phoenix-heavy systems

## What Belongs in Long-Lived Process State

Good candidates:

- coordination state that must survive across messages
- subscription lifecycle state
- controlled caches with explicit invalidation
- bounded in-memory workflow state

Bad candidates:

- arbitrary request payload history
- large domain aggregates that belong in storage
- mixed concerns from unrelated workflows
- hidden global state used as a service locator

## Bad vs Good

```text
❌ BAD
A GenServer owns request validation, business rules, DB calls, retry logic, and HTTP response mapping.

✅ GOOD
Transport validates and maps requests.
A use-case module orchestrates the workflow.
Processes are introduced only where state, supervision, or serialization are truly needed.
```

## Review Questions

- is this process boundary expressing domain ownership or just implementation convenience?
- which state truly needs to live in a GenServer?
- what blast radius does this design create?
