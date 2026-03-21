# React Frontend Architecture Decision Framework

## Purpose

Use this guide to choose the smallest frontend architecture that stays maintainable under product growth, async complexity, and multi-team ownership.

## Decision Axes

- route complexity
- server-state complexity
- mutation frequency
- form complexity
- table/reporting needs
- multi-team ownership
- SSR vs SPA vs hybrid requirements

## Choose a Simple SPA Structure When

- routing is shallow
- server data is limited
- forms are straightforward
- one team owns most features

## Increase Boundary Strength When

- routes own heavy loaders and URL state
- optimistic updates and invalidation rules get complex
- data tables, filters, and mutations interact heavily
- multiple teams change features concurrently

## Bad vs Good

```text
❌ BAD
Use folders and hooks inconsistently because the app is still "small".

✅ GOOD
Adopt predictable boundaries before complexity becomes organic chaos.
```

## Review Questions

- where should URL state live?
- where should server state live?
- what structure reduces merge conflict and accidental coupling?
