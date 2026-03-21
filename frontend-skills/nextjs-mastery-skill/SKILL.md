---
name: nextjs-principal-engineer
description: |
  Principal/Senior-level Next.js playbook for App Router architecture, server components, caching, revalidation, route handlers, auth, observability, performance, and production operations.
  Use when: building or reviewing Next.js applications, designing rendering strategy, structuring server/client boundaries, hardening data fetching, improving performance, or preparing production deployment.
---

# Next.js Mastery (Senior → Principal)

## Operate

- Confirm Next.js version, App Router usage, runtime targets, rendering requirements, auth model, cache/revalidation expectations, mutation flows, and deployment platform.
- Separate server components, client components, route handlers, and external backend dependencies intentionally.
- Optimize for rendering correctness, predictable cache behavior, and operability under production traffic.

## Default Standards

- choose server vs client boundaries deliberately
- keep cache and revalidation behavior explicit
- avoid accidental client-heavy architecture inside App Router apps
- treat auth, mutations, and route handlers as production boundaries
- instrument slow routes, cache misses, and runtime-specific regressions

## References

- App Router architecture: [references/app-router-architecture.md](references/app-router-architecture.md)
- Server and client component boundaries: [references/server-client-boundaries.md](references/server-client-boundaries.md)
- Caching and revalidation strategy: [references/caching-and-revalidation.md](references/caching-and-revalidation.md)
- Route handlers and backend integration: [references/route-handlers-and-backend-integration.md](references/route-handlers-and-backend-integration.md)
- Performance and streaming: [references/performance-and-streaming.md](references/performance-and-streaming.md)
- Incidents, debugging, and production operations: [references/incidents-debugging-and-ops.md](references/incidents-debugging-and-ops.md)
