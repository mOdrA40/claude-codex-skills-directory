# Multi-Tenant Platform Governance

## Rules

- Shared meshes need policy about namespaces, egress, authz, and traffic features.
- Tenant autonomy should not create global policy chaos.
- Platform teams should publish safe defaults and constrained escape hatches.
- Governance should reduce blast radius while preserving team velocity.

## Governance Guidance

- Standardize which resources app teams may own directly.
- Track high-risk objects such as gateways, authz policies, and global traffic rules.
- Make exception paths explicit, reviewable, and time-bounded where possible.
- Keep ownership clear for certificates, gateways, and control plane config.

## Principal Review Lens

- Which tenant can create the biggest mesh incident today?
- Are unsafe traffic features too easy to enable?
- What policy gap is causing recurring platform toil?
- Is governance helping scale or merely pushing teams around?
