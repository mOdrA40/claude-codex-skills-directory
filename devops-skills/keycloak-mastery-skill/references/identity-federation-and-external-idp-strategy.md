# Identity Federation and External IdP Strategy

## Rules

- Federation introduces dependency, trust, and failure complexity.
- External IdP integration should have explicit ownership and fallback thinking.
- User lifecycle, attribute mapping, and group sync need disciplined governance.
- Trusting upstream identity claims should never be casual.

## Practical Guidance

- Document how login behaves when an upstream IdP is slow or unavailable.
- Standardize mapping, provisioning, and deprovisioning rules.
- Keep break-glass access separate from federated dependency chains.
- Review which identity sources are authoritative for which attributes.

## Principal Review Lens

- Which upstream IdP dependency has the highest blast radius?
- Can the team operate safely when federation degrades?
- What attribute mapping is most likely to create access mistakes?
- Are we clear about who owns identity truth for each workflow?
