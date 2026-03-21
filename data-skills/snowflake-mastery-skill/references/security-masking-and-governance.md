# Security, Masking, and Governance

## Rules

- Access control, masking, and data classification should be explicit and reviewable.
- Sensitive data requires stronger governance than ordinary analytics convenience.
- Governance should map to real legal, compliance, and business risk.
- Role sprawl is operational and security debt.

## Practical Guidance

- Standardize role design and privilege review.
- Apply masking and row/column controls where they reduce real risk.
- Keep access workflows fast enough to avoid shadow-data patterns.
- Audit high-risk objects and privilege grants regularly.

## Governance Heuristics

### Fast access and strong governance must coexist

If secure access takes too long or feels too opaque, teams will create shadow exports, ad hoc copies, or workaround paths that reduce trust far more than the original policy intended.

### Masking policy should reflect real data use

The best masking strategy depends on how data is actually consumed, by whom, and for what decision scope—not only on abstract classification labels.

### Role design is an operating model choice

Roles should make least privilege, auditability, and business ownership understandable enough that access reviews remain practical at scale.

## Common Failure Modes

### Governance theater

Policies exist in principle, but real privilege paths, broad roles, or copied access patterns make the effective posture much weaker.

### Secure-path friction creating shadow behavior

Teams bypass governed access because official workflows are too slow, unclear, or mismatched to operational reality.

### Masking without consumer understanding

Data is technically masked, but consumers and owners do not share a clear understanding of what remains sensitive or what decisions the masked form still enables.

## Principal Review Lens

- Which data set has the weakest real protection today?
- Are masking policies aligned with actual consumption patterns?
- What access path is too broad or too opaque?
- Which governance gap most threatens platform trust?
- Which convenience path is currently eroding governance more than teams admit?
