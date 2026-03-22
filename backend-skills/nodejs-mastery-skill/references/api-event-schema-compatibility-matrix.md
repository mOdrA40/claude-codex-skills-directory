# API and Event Schema Compatibility Matrix for Node.js Services

## Purpose

This matrix helps agents reason about whether changes are backward-compatible, forward-compatible, or rollout-dangerous.

## Safe-by-Default Changes

- adding optional response fields
- accepting additional optional request fields
- adding new event fields that old consumers ignore safely

## Risky Changes

- removing fields used by old readers
- changing field meaning without versioning
- tightening validation while old producers still emit previous shapes
- changing ordering assumptions for events

## Agent Questions

- can old and new producers/consumers coexist?
- is this change additive or breaking?
- does rollback remain safe after this change?
- are queues, webhooks, and async workers included in compatibility thinking?
