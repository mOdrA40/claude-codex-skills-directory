# Elysia Plugin Risk Management

## Purpose

Plugin-heavy systems can become operationally opaque. Elysia productivity is strongest when plugin boundaries remain understandable during debugging and incidents.

## Rules

- keep plugin scope narrow and named clearly
- separate foundational plugins from feature plugins
- avoid magic behavior that rewrites request/response semantics invisibly
- ensure plugin-added context is explicit in types and runtime behavior

## Review Questions

- can the team explain plugin execution order confidently?
- are auth, validation, and observability plugins clearly separated?
- what breaks if one plugin is removed or reordered?
