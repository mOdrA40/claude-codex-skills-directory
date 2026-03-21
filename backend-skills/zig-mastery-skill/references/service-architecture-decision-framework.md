# Service Architecture Decision Framework for Zig

## Principle

Zig gives strong control, but architectural discipline still matters more than low-level power. Use Zig where explicitness buys real operational or performance value.

## Choose Zig When

- allocator control materially affects latency or memory behavior
- FFI or systems integration is central
- predictable binary/runtime behavior is important
- the team can support explicit systems-level design discipline

## Avoid Overreach

Do not use Zig just to prove technical taste if:

- the service is ordinary CRUD with low systems pressure
- framework maturity matters more than control
- the team cannot support explicit memory and contract discipline yet

## Review Questions

- what real risk does Zig reduce here?
- what human cost does the team pay for this control?
- is the service shape actually systems-sensitive?
