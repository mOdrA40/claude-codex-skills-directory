# Outage Triage: First 15 Minutes in Node.js Services

## Goal

Contain blast radius before chasing elegance.

## First Questions

- did a deploy just happen?
- is one dependency failing or is the service itself saturated?
- are errors concentrated on one route, tenant, or worker class?
- is queue lag rising?
- is event loop delay increasing?

## Immediate Containment Options

- stop or slow rollout
- fail readiness on bad instances
- shed low-value traffic
- disable optional fan-out or enrichment
- pause problematic consumers if duplicate or toxic work is amplifying failure

## What Not To Do

- do not add retries blindly
- do not restart everything without understanding the bottleneck
- do not assume all 500s are app bugs
- do not change schema or infrastructure mid-incident unless required for containment

## Agent Heuristic

If the user asks to fix an outage quickly, propose containment first, root cause second, and cleanup third.
