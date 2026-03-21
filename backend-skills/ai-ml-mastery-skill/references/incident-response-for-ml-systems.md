# Incident Response for ML Systems

## Principle

ML incidents are not always outages. Silent degradation, stale features, retrieval drift, prompt abuse, and runaway cost are all production incidents.

## Common Incident Classes

- latency spikes in inference
- model artifact loading failure
- feature freshness degradation
- retrieval quality collapse
- drift beyond safe thresholds
- prompt abuse or adversarial input patterns
- token or cost explosion

## First Questions

- is the issue infrastructure, feature pipeline, model behavior, or retrieval quality?
- should the model be rolled back, disabled, or put into degraded mode?
- what user segments are most at risk right now?

## Review Questions

- can operators disable or route around the model quickly?
- what signal tells you to fall back?
- is the incident a hard failure or silent decision-quality failure?
