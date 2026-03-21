# Policy Architecture and Risk Boundaries (OPA Gatekeeper)

## Rules

- Policy should target the highest-value platform risks first.
- Enforcement boundaries should reflect cluster criticality, tenancy, and support maturity.
- Not every opinion belongs in admission control.
- Policy architecture should remain explainable to both platform and application teams.

## Practical Guidance

- Separate security-critical, reliability-critical, and hygiene policies deliberately.
- Identify which policies should block admission and which should only audit.
- Keep cluster, namespace, and workload scope explicit.
- Avoid policy catalogs that nobody can maintain or reason about.

## Principal Review Lens

- Which policy class most reduces real risk?
- Are we enforcing useful guardrails or just centralizing taste?
- What policy belongs outside admission control entirely?
- Which boundary is too broad for safe enforcement?
