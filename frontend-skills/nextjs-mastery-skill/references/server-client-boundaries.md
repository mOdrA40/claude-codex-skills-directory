# Server and Client Component Boundaries

## Principle

The biggest architectural decision in modern Next.js is not only component design. It is deciding where execution happens and what data crosses the boundary.

## Heuristics

### Prefer server components when

- content can be rendered from server-fetched data
- no browser-only API is needed
- personalization can still be resolved safely on the server
- you want lower client bundle cost

### Use client components when

- browser APIs are required
- interactive local state is central
- event handlers and user-driven transitions dominate the component's value

## Boundary Design Questions

Before choosing the boundary, ask:

- does this component need browser execution or just interactive composition nearby?
- how much serialized data crosses the server-client boundary?
- what bundle cost does this choice impose on the route?
- what testing and debugging burden does this boundary create?

## Common Anti-Patterns

### Client wrapper inflation

A small interactive requirement causes large route segments to become client-rendered, increasing bundle size and hiding server-first design opportunities.

### Boundary by convenience

Teams choose client components because they feel easier to debug locally, even when the user pays with slower delivery and more hydration work.

## Failure Modes

- passing too much data across boundaries
- leaking server assumptions into client code
- using client components for convenience instead of necessity
- hiding performance regressions behind innocuous-looking client wrappers

### Serialization surprise

Data shapes look reasonable in development, but crossing the boundary repeatedly becomes expensive or awkward under real route complexity.

## Review Questions

- what state truly needs to exist in the browser?
- what payload size or serialization cost crosses the boundary?
- is this client component small and focused or acting as an architectural escape hatch?
- what route performance or debugging problem is this boundary choice likely to create later?
