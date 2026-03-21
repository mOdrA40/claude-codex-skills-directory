# Performance Tuning and Traffic Shaping

## Rules

- Tune based on traffic shape, payload size, connection behavior, and upstream characteristics.
- Edge buffering, timeouts, keepalives, and retries can help or hurt depending on workload.
- Rate limiting and traffic shaping should reflect product priorities and abuse patterns.
- Benchmark with realistic request distributions, not happy-path demos.

## Common Failure Modes

- Queueing and timeouts masking upstream slowness.
- Retry behavior amplifying overload.
- Body limits or buffering settings breaking uploads or streaming unexpectedly.
- One tenant or path consuming disproportionate edge resources.

## Principal Review Lens

- Which ingress setting most amplifies bad upstream behavior?
- Are timeouts protecting users or hiding deeper issues?
- What traffic policy changes behavior under peak load the most?
- Which workload deserves different ingress treatment than the default?
