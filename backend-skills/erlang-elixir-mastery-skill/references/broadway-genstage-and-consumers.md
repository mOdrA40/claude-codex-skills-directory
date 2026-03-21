# Broadway, GenStage, and Consumer Pipelines

## Principle

BEAM concurrency makes consumers easy to build, but easy concurrency is not the same as safe queue semantics. Throughput, ordering, retries, and backpressure still need explicit design.

## Rules

- decide whether ordering matters
- define retry and dead-letter semantics
- make concurrency and demand settings intentional
- expose lag, age, throughput, and failure-class metrics
- keep poison-message handling explicit

## Review Questions

- what happens when one event is malformed?
- can one slow external dependency stall the whole pipeline?
- where is backpressure applied first?
- can duplicate delivery happen and is it safe?
