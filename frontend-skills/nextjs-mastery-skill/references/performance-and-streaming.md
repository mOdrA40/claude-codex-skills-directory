# Performance and Streaming in Next.js

## Principle

Performance work in Next.js should look at route cost, bundle cost, data dependency shape, and streaming value. Not every slow page is a bundle problem.

## Performance Model

When users say a route is slow, they may mean very different things:

- the server took too long to start responding
- meaningful content appeared late
- the page painted quickly but stayed non-interactive
- route transition felt blocked
- stale or half-loaded content reduced trust even when paint was fast

Treat these as separate failure classes instead of one generic performance problem.

## Investigation Order

1. identify whether the user pain is first byte, first paint, hydration, interactivity, or route transition latency
2. inspect route-level data dependencies
3. inspect client bundle boundaries
4. inspect streaming and suspense placement
5. inspect cache assumptions and mutation aftermath

## Streaming Heuristics

### Stream only when it improves perceived progress

Streaming is valuable when users can do something meaningful earlier or at least build trust that progress is happening. If the streamed shell is mostly decorative, the complexity may not be paying for itself.

### Keep suspense boundaries aligned with user value

Good suspense boundaries separate:

- above-the-fold essentials
- secondary panels
- enhancement-only regions
- expensive, non-blocking recommendations or analytics surfaces

Bad suspense boundaries simply mirror file structure and give users no meaningful progress advantage.

### Hydration cost still matters after streaming

Streaming can make the page appear fast while the real interaction bottleneck remains hidden behind client subtree hydration.

## Common Failure Modes

- large client subtrees hiding in server-first routes
- suspense boundaries that do not reduce user-visible wait meaningfully
- route transitions blocked by unnecessary client work
- apparent performance wins that create stale-data confusion

### Route architecture fighting streaming

The route depends on a few oversized blocking dependencies, so streaming exists in theory but not in practice.

### Performance without trust

The route feels visually fast, but users repeatedly see stale, inconsistent, or partially refreshed content after mutations.

## Incident Patterns

### Fast locally, slow in production

Often caused by:

- different traffic shape
- real cache misses
- production-only auth or middleware behavior
- route handlers or backend integration latency not visible during local development

### Streaming changed the UX but not the outcome

Users still wait for the same meaningful interaction point, so the team improved paint timing but not product experience.

## Review Questions

- what user-visible waiting time is streaming supposed to reduce?
- what data dependency can be deferred safely?
- is the problem route architecture, bundle architecture, or backend latency?
- which performance win matters most for this route: first byte, first meaningful content, or first useful interaction?
