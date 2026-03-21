# Performance and Packaging in Tauri

## Principle

Tauri performance depends on frontend route cost, Rust command efficiency, local IO behavior, and packaging/update choices together.

## Common Failure Modes

- optimizing bundle size while local IO or command cost dominates
- packaging that works in development but is painful in real-user upgrade flows
- slow startup from unnecessary bootstrap work on both frontend and native sides

## Review Questions

- what dominates startup time?
- what packaging or distribution choice increases operational risk?
- which desktop workflow feels slowest to users and why?
