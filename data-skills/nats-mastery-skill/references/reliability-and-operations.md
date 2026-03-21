# Reliability and Operations (NATS)

## Operational Defaults

- Monitor connection churn, message rates, JetStream health, consumer lag, redelivery, and cluster route state.
- Keep upgrades and topology changes staged and reversible.
- Separate ephemeral messaging issues from durable-stream issues quickly.
- Document ownership of accounts, streams, and critical subjects.

## Run-the-System Thinking

- NATS simplicity helps only if subject and account governance remain sane.
- Capacity planning should include reconnect storms and replay events.
- On-call needs clear guidance on what is safe to purge, replay, or isolate.
- Operational maturity depends on boring topology and explicit semantics.

## Principal Review Lens

- Which signal tells you a bad platform day is starting?
- What action isolates customer pain fastest?
- Can the team explain which traffic is ephemeral versus durable?
- Are we running a disciplined messaging platform or an ad-hoc bus?
