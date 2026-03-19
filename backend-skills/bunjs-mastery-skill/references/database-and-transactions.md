# Database and Transactions (Bun Services)

Most production bugs around data are not caused by missing ORM features. They come from weak transaction boundaries, hidden retries, and unclear ownership of invariants.

## Defaults

- Keep write invariants explicit.
- Use transactions only where they protect a real invariant.
- Bound pool size and acquisition time.
- Prefer unique constraints over wishful application logic.
- Keep slow analytical queries away from hot request paths.

## Transaction Rules

Use transactions for:

- multi-step writes that must succeed or fail together,
- balance/inventory/quota style updates,
- persisting outbox records with state changes,
- read-modify-write flows needing atomicity.

Avoid transactions that cover network calls or long-running business workflows.

## Query Design

- Review queries for N+1 patterns.
- Make pagination explicit and stable.
- Keep indexes aligned with hottest read paths.
- Prefer deterministic ordering in user-visible lists.

## Reliability Guardrails

- Time out acquisition and slow queries.
- Emit metrics for query duration and pool pressure.
- Be explicit about retry behavior on transient DB failures.
- Distinguish conflict from internal failure in public error mapping.

## Principal Review Lens

Ask:

- Which business invariant is protected here?
- What happens under duplicate request delivery?
- Which queries dominate p95 latency?
- If the database is slow, how does the service degrade?
