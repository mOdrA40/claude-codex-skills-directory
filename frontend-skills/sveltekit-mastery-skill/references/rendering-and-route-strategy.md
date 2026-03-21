# Rendering and Route Strategy in SvelteKit

## Principle

SvelteKit is at its best when route behavior is chosen intentionally. SSR, CSR, prerendering, and enhanced forms should reflect product needs, not default habit.

## Route Strategy Model

Each route should have an explicit answer for:

- who needs it and when
- whether SEO matters materially
- how fresh the data must be
- how interactive the surface is
- what degraded behavior is acceptable

This is a product and operational choice, not only a rendering preference.

## Decide Per Route

Consider:

- SEO criticality
- personalization level
- freshness requirements
- interaction density
- deployment constraints

## Common Failure Modes

- globally applying one render strategy that fits only half the app
- hiding personalization problems behind simplistic SSR assumptions
- overusing client-only behavior because it feels easy

### Route strategy drift

The app started with clean route intentions, but later changes introduced conflicting assumptions about freshness, personalization, and client behavior.

### Strategy by framework default

The team accepts the easiest mode for the framework instead of defining what the route actually needs for user value and operability.

## Review Questions

- should this route be server-rendered, prerendered, or mostly client-driven?
- what is the cost of being wrong for this route class?
- what route behavior becomes hard to debug in production?
- which route class in the product most needs an explicit rendering policy today?
