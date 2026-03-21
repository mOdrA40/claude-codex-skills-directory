# Networking and Network Policy

## Rules

- Network policy should express least privilege, not afterthought hardening.
- Service discovery, DNS, and east-west traffic deserve explicit design.
- Debuggability matters as much as policy strictness.
- CNI behavior and limits must be understood by platform owners.

## Principal Review Lens

- Which workload can talk to too much today?
- Is policy understandable enough to debug safely?
- What network assumption breaks during node churn?
