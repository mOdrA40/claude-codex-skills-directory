# Streaming and Backpressure in Node.js

## Principle

Node.js is strong at streaming, but streaming without backpressure discipline becomes a latency, memory, and reliability problem.

## Use Streaming When

- payloads are large
- end-to-end buffering is wasteful
- partial delivery improves latency
- upstream and downstream can flow-control correctly

## Do Not Assume

Streaming is not automatically cheap. It can still fail through:

- slow consumers
- uncontrolled buffering
- compression overhead
- per-chunk expensive transforms
- hidden retries or reconnect loops

## Bad vs Good

```text
❌ BAD
The service reads the whole upstream response into memory, transforms it, then sends it.

✅ GOOD
The service streams with explicit size, timeout, and cancellation behavior.
```

## Backpressure Questions

- Which component applies flow control first?
- What happens if the client is slow?
- What happens if upstream is fast but downstream is stalled?
- Is buffering bounded?
- Are disconnects and cancellations propagated?

## Operational Guidance

- measure stream duration and bytes transferred
- distinguish upstream stall vs downstream stall
- cancel upstream work when the client disconnects
- avoid mixing streaming paths with blocking CPU transforms on the main thread
