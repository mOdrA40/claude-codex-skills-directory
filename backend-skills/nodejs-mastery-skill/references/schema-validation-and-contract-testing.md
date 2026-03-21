# Schema Validation and Contract Testing

## Purpose

Validation prevents bad input from crossing boundaries. Contract testing prevents services from lying to each other about what a payload means.

## Rules

- validate input at the edge
- validate outbound dependency responses when trust is low or failure cost is high
- keep schemas version-aware when contracts evolve
- test compatibility for critical service-to-service payloads and events

## Bad vs Good

```text
❌ BAD
The handler trusts internal clients and only validates public API traffic.

✅ GOOD
Critical boundaries validate both inbound and outbound contracts where failure impact justifies it.
```

## Review Questions

- what happens if the downstream response shape changes silently?
- which contracts are operationally critical enough to deserve compatibility tests?
