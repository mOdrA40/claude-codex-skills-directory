# Tracing Model and Span Design (OpenTelemetry)

## Rules

- Create spans around meaningful latency boundaries, not every function call.
- Span names should reflect stable business or technical operations.
- Attributes must be bounded, useful, and aligned with semantic conventions.
- Events should add forensic value, not duplicate logs blindly.

## Design Heuristics

- Root spans should represent externally meaningful work.
- Child spans should expose dependency timing, internal queueing, or critical work phases.
- Errors should be represented consistently across languages and runtimes.
- Links and async trace structures should model causality honestly.

## Principal Review Lens

- Which spans actually help during a customer-facing incident?
- Are we tracing code shape instead of system behavior?
- Which attributes will become expensive noise at scale?
- What trace gap hurts dependency reasoning most today?
