# API Evolution and Versioning in Bun Services

## Principle

APIs are contracts first, implementation details second. Fast iteration does not excuse unsafe compatibility changes.

## Rules

- prefer additive change before version splits
- deprecate explicitly
- distinguish schema compatibility from semantic compatibility
- keep error contracts stable across minor changes
- coordinate queue events and webhooks with the same rigor as HTTP payloads

## Review Questions

- can old and new clients coexist during rollout?
- are event consumers coupled to a field that is about to change?
- is a semantic break being disguised as a compatible payload change?
