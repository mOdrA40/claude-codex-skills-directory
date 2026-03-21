# Contract Validation and Response Mapping in Bun APIs

## Principle

Fast runtimes still need boring contracts. In Bun APIs, request validation and response mapping should be stable, explicit, and operationally observable.

## Rules

- validate request bodies, params, and headers intentionally
- map domain errors to stable HTTP payloads
- avoid leaking raw DB or runtime errors
- validate critical downstream response shapes when failure cost is high

## Review Questions

- can clients rely on stable error contracts?
- do handlers distinguish invalid input from dependency failure clearly?
- will downstream schema drift be caught early?
