# xDS, Config Delivery, and Governance

## Rules

- Dynamic config delivery is a platform control plane concern, not mere convenience.
- Governance should define who can change what and how config is reviewed.
- Safe rollout, versioning, and rollback of config are essential.
- Avoid dynamic power without operational guardrails.

## Governance Guidance

- Track ownership of route config, clusters, listeners, and shared policies.
- Keep config diffs reviewable and semantically meaningful.
- Test config at the edges of routing and failure behavior.
- Protect high-blast-radius config paths with stronger process.

## Principal Review Lens

- Which xDS path has the weakest review discipline?
- Can one misconfig affect too many services at once?
- Are we treating config delivery as code or as live surgery?
- What governance rule would most reduce proxy incidents?
