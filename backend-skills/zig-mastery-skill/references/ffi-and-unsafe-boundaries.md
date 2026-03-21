# Zig FFI and Unsafe Boundaries

## Purpose

Many Zig backends earn their keep by integrating with C libraries, system APIs, or specialized native components. That power becomes a production liability if FFI boundaries are not isolated and reviewed as high-risk code.

## Principles

- Keep unsafe boundaries narrow.
- Convert foreign APIs into Zig-friendly contracts once.
- Validate lengths, nullability, ownership, and lifetime assumptions explicitly.
- Never let raw foreign handles leak everywhere in the codebase.

## Boundary Strategy

Treat FFI code as an adapter layer:

- application/domain code should depend on safe Zig interfaces
- adapters own handle management and translation
- memory ownership must be documented near the FFI boundary

## Bad vs Good: FFI Leakage

```text
❌ BAD
Application code calls foreign functions directly across many files.

✅ GOOD
A small adapter module wraps the foreign API and exposes typed Zig functions.
```

## Risk Checklist

- Who owns the returned buffer?
- Is the pointer nullable?
- Who frees it?
- Is the foreign library thread-safe?
- Can the library block forever?
- Are there global initialization requirements?
- What happens on partial initialization failure?

## Operational Guidance

- Log library init failures distinctly.
- Treat FFI crashes as security- and reliability-sensitive incidents.
- Add smoke tests that exercise adapter initialization in CI.
- Prefer restart-safe initialization paths.
