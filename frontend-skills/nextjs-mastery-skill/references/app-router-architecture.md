# App Router Architecture in Next.js

## Principle

App Router is powerful because it gives multiple execution models in one system. That is also why teams create confusion when they do not define ownership boundaries clearly.

## Architecture Goals

A strong Next.js architecture should answer:

- which routes are primarily server-rendered
- which components truly need client execution
- where auth and personalization are resolved
- where data fetching is owned
- which mutations belong in route handlers, server actions, or external backend services

## Boundary Model

Separate:

- route segments and layouts
- server components
- client components
- route handlers or server actions
- external service clients
- cache and revalidation policy

## Route Ownership Heuristics

### Layouts should own shared shell concerns

Examples:

- navigation shell
- stable shared data needed across child routes
- cross-route UI scaffolding

They should not quietly become the place where unrelated feature logic accumulates.

### Pages should own route-specific rendering intent

Examples:

- route-specific data composition
- page-level user journey shaping
- page-local fallback and error strategy

### Backend integration layers should own durable service contracts

Do not hide cross-service orchestration deep inside the component tree.

## Common Failure Modes

### Client sprawl inside App Router

Teams mark large parts of the tree with `use client` because one child needs interactivity. This throws away the main architectural value of the model.

### Layout ownership confusion

Too much data loading or business logic ends up in shared layouts, making route-specific behavior hard to reason about.

### Backend logic hidden in UI files

Fetching, mutation orchestration, and error mapping drift into component trees instead of living in clearer boundaries.

### Segment nesting without ownership clarity

The route tree grows, but nobody can explain why a concern belongs to one segment rather than another.

## Bad vs Good

```text
❌ BAD
Turn most of the route tree into client components because it feels easier to reason about.

✅ GOOD
Keep server-first boundaries by default and isolate client interactivity to the smallest meaningful areas.
```

## Review Questions

- what is the minimum interactive subtree this route needs?
- which route concerns belong in layout vs page vs shared backend integration layer?
- where will operators look first when route performance regresses?
- which route segment would be hardest to change safely six months from now?
