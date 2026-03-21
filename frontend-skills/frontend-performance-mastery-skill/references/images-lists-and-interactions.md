# Images, Lists, and Interaction Performance

## Principle

Most real frontend performance pain comes from repetitive UI patterns: long lists, media-heavy surfaces, and interaction loops that trigger too much work.

## Common Failure Modes

- list virtualization omitted or misapplied
- images too large or decoded at the wrong time
- interactions triggering expensive state updates across broad regions

## Review Questions

- what repeated UI pattern dominates cost?
- is media handling proportional to device constraints?
- what interaction causes the worst tail-latency for users?
