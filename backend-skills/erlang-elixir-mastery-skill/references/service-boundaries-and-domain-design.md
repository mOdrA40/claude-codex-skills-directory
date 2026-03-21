# Service Boundaries and Domain Design on the BEAM

## Principle

The BEAM makes concurrency natural, but concurrency is not architecture. Domain boundaries, message design, and ownership still need to be explicit.

## Rules

- keep domain boundaries coherent before choosing process topology
- do not replace design with more processes
- decide which responsibilities deserve long-lived process state and which do not
- keep transport concerns at the edge even in Phoenix-heavy systems

## Review Questions

- is this process boundary expressing domain ownership or just implementation convenience?
- which state truly needs to live in a GenServer?
- what blast radius does this design create?
