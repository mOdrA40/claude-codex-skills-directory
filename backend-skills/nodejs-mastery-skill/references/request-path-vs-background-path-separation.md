# Request Path vs Background Path Separation in Node.js Services

## Core Principle

Do not keep expensive, failure-prone, or slow fan-out work on the request path when correctness allows durable async handling.

## Good Candidates for Background Work

- email or webhook fan-out
- expensive enrichment not required for immediate user correctness
- large imports or batch sync
- slow third-party publishing

## Keep on Request Path When

- the user needs the result immediately for correctness
- durable async workflow is unavailable or unsafe
- partial async handling would create confusing semantics

## Agent Questions

- what is the user-visible correctness boundary?
- will async handling improve or harm operability?
- does moving this work async require idempotency, queues, and replay safety?
