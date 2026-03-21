# Performance and Route Delivery in Remix

## Principle

Performance in Remix depends on route hierarchy, data dependency shape, and mutation aftermath as much as on asset delivery.

## Route Delivery Model

Users experience Remix performance through:

- initial route load
- nested route composition cost
- loader dependency shape
- mutation aftermath and refetch behavior
- navigation stability between routes

If teams focus only on bundle metrics, they often miss the real source of user waiting.

## Common Failure Modes

- nested routes fetching inefficiently
- route transitions blocked by avoidable work
- performance tuning that ignores data ownership problems

### Hierarchy tax

Nested routes look elegant but create hidden waiting because data ownership and dependency order were never designed intentionally.

### Mutation-triggered churn

After actions complete, too much of the route hierarchy refreshes or becomes temporarily untrustworthy.

## Investigation Heuristics

### Identify the slow boundary

Ask whether the main cost lives in:

- parent route loaders
- child route loaders
- navigation-induced revalidation
- mutation aftermath

### Tie route performance to ownership

If no one can explain who owns data for a slow route segment, optimization work will likely chase symptoms.

## Review Questions

- what route boundary causes the most waiting?
- what dependency should be deferred or narrowed?
- where is route hierarchy harming delivery?
- what route performance issue is actually an ownership issue in disguise?
