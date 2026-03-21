# Streaming and Backpressure in Rust Services

## Purpose

Rust can model backpressure well, but that advantage disappears if streams are treated as just another async abstraction without operational limits.

## Rules

- bound stream buffers intentionally
- propagate cancellation on disconnect or timeout
- distinguish producer slowness from consumer slowness
- avoid silent memory growth from buffered streams or channels

## Review Questions

- where is flow control applied first?
- what happens when downstream slows dramatically?
- does buffered work have a hard cap?
