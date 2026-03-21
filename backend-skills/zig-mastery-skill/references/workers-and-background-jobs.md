# Workers and Background Jobs in Zig

## Principle

Background work is where ownership bugs, shutdown bugs, and retry bugs become operational incidents. Zig's explicit style should be used to make worker lifecycles obvious.

## Rules

- define who starts and stops workers
- bound concurrency and queue depth
- make retry semantics explicit
- classify terminal vs transient failure
- expose metrics for lag, age, throughput, and failure rate

## Review Questions

- can one stuck job stall the worker pool?
- how does the service drain on shutdown?
- what happens when downstream is slow for a long time?
