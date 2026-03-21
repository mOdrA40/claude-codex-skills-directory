# Solid Architecture Decision Framework

## Purpose

Solid rewards simple reactive architecture, but teams still need explicit guidance for when to keep logic local, when to use stores, and when to add framework-level complexity.

## Rules

- keep signals local when ownership is local
- use stores when shape is shared and nested updates matter
- use resources for async boundaries, not generic state containers
- keep server and client concerns distinct in SolidStart

## Decision Heuristics

### Use local signals when

- one component or a very small subtree owns the state
- the state is ephemeral and interaction-driven
- you do not need persistence or cross-feature sharing

### Use stores when

- state shape is nested and shared intentionally
- multiple updates must remain coherent
- path-based updates improve clarity

### Use resources when

- the state is fundamentally async and fetch-driven
- suspense/error boundaries are part of the desired UX
- the source of truth is not local interaction state

## Common Failure Modes

### Store-first thinking

Teams import habits from other frameworks and create broad stores when local signals or resources would be simpler.

### Effect-heavy design

Effects start doing derived-state work that should have been modeled as memos or clearer reactive dependencies.

### Mixed server/client assumptions

SolidStart code blurs server-only and client-only logic, producing hydration or execution surprises.

## Review Questions

- is this state local, shared, server-owned, or derived?
- is a store being used because it is needed or because it feels familiar from other frameworks?
- would a simpler signal/memo design be clearer?
- what boundary is carrying unnecessary complexity just because the framework is flexible?
