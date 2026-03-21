# Publisher Confirms and Delivery

## Rules

- Publisher confirms are baseline for reliable publishing.
- Mandatory routing and unroutable handling should be explicit.
- Message durability and queue durability must align.
- Publish-side idempotency still matters.

## Principal Review Lens

- What does a successful publish actually guarantee here?
- How are unroutable messages surfaced and handled?
- Where can duplicates still appear despite confirms?
