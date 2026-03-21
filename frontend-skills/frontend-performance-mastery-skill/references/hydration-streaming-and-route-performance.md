# Hydration, Streaming, and Route Performance

## Principle

Modern frontend route performance depends on when content appears, when it becomes interactive, and how much client work is still hidden behind the initial paint.

## Route Performance Model

Users may experience route performance through several separate lenses:

- how quickly meaningful content appears
- how quickly the route becomes interactive
- how stable the UI feels during hydration
- how predictable navigation feels between routes

Treating these as one number leads to misleading optimization work.

## Common Failure Modes

- fast server output with slow hydration reality
- streaming that does not improve perceived progress
- route architectures that block on non-critical dependencies

### Hydration optimism

The team celebrates faster server response while users still wait on large client islands or heavy interaction bootstrap.

### Streamed shell with weak product value

The route appears faster but users still cannot read, trust, or act on anything meaningful earlier.

## Investigation Heuristics

### Map the true user wait

Ask whether the delay is in:

- first useful content
- first trusted content
- first actionable interaction
- post-navigation route stability

### Tie route structure to dependency structure

If a route blocks on too many critical dependencies, hydration and streaming optimizations may only mask architectural problems.

## Review Questions

- what is the user waiting for first?
- which dependency can be deferred?
- where is apparent speed masking real interactivity cost?
- what route currently looks fast in metrics but still feels slow to real users?
