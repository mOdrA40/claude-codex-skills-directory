# Debugging and Production Operations in Astro

## Principle

Production Astro incidents often involve content, build, hydration, or deployment pipeline behavior rather than classic client-heavy runtime bugs.

## Incident Classes

- content rendering failures
- preview/publish mismatches
- hydration regressions on interactive islands
- route or image pipeline delivery regressions

## Review Questions

- is the issue content-model, build-time, hydration-time, or delivery-time?
- what release or content change preceded the incident?
- how can blast radius be reduced while content remains available?
