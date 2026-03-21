# Retries, DLQ, and Backoff

## Rules

- Retry policy must avoid hot loops and silent backlog growth.
- DLQ ownership and replay workflow must be explicit.
- Exponential backoff should reflect business urgency and system safety.
- Poison messages need quarantine, not wishful retries.

## Principal Review Lens

- Who owns DLQ triage?
- What retry setting turns one failure into a platform incident?
- How is replay made safe for side effects?
