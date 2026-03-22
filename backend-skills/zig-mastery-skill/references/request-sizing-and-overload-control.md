# Request Sizing and Overload Control in Zig Services

## Purpose

Zig services often win by being explicit. Overload control should be equally explicit. If request size, concurrency, and queue growth are left implicit, the service will eventually let memory pressure and dependency saturation choose the failure mode.

## Design Goals

A production-safe Zig service should define:

- maximum request body size
- maximum parsing depth or record count
- maximum concurrent expensive operations
- maximum queue depth for background work
- behavior when those limits are exceeded

## Overload Policies

Choose one or more intentionally:

- reject early with clear status or error
- queue with bounded depth
- degrade optional work
- rate-limit by tenant or caller
- shed load when dependencies are saturated

## Review Questions

- what is the largest request the service will accept?
- which layer enforces that limit?
- can one caller exhaust memory through valid-but-huge payloads?
- what happens when downstream latency increases 10x?
- which work is critical enough to preserve and which can be dropped?

## Principal Heuristics

- Prefer failing fast and observably over slow collapse.
- Tie overload control to SLOs and business criticality, not guesswork.
- If the only backpressure mechanism is process death, the service is not production-ready.
