# Route Rules and Rendering Strategy in Nuxt

## Purpose

Choosing SSR, SSG, ISR-style caching, or client-heavy rendering should be a product and operational decision, not a default habit.

## Rules

- choose rendering strategy per route class
- keep SEO-critical pages different from app-shell dashboards when needed
- align cache TTL and revalidation with data freshness requirements
- make auth and personalization constraints explicit before choosing render mode

## Review Questions

- should this route be pre-rendered, server-rendered, or client-driven?
- what freshness guarantees matter?
- how expensive is hydration for this route?
