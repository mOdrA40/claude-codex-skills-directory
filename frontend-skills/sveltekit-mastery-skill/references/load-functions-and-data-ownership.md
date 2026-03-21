# Load Functions and Data Ownership in SvelteKit

## Principle

Load functions make data flow elegant when ownership is clear and confusing when everything tries to fetch everywhere.

## Rules

- know which data belongs to route loading vs local interaction
- avoid duplicated fetch logic between universal and server-only boundaries
- keep error and redirect behavior explicit
- treat data shape and freshness as product decisions, not only implementation details

## Ownership Heuristics

### Route-owned data

Good candidates include:

- route-blocking data
- SEO or first-render critical data
- data whose failure should affect route-level rendering decisions

### Local interaction data

Good candidates include:

- ephemeral search input
- transient filter state
- client-only enhancement state

### Shared layout data

Use carefully. Shared layout data should support true shared shell behavior, not become a convenience bucket for unrelated fetching.

## Failure Modes

- parent and child routes refetching without intentional design
- route data and client state fighting over ownership
- redirects hidden inside data-loading logic without clear operational visibility

### Data scope inflation

One layout or parent route loads broad data because it is convenient, and now child routes inherit hidden coupling and slower delivery.

## Review Questions

- who owns this fetch?
- what happens on failure or stale data?
- what route segment should be allowed to redirect?
- which route load would be hardest to simplify later because it owns too much today?
