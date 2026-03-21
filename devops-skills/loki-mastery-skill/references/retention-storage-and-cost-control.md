# Retention, Storage, and Cost Control

## Rules

- Retention should match compliance, debugging, and audit value explicitly.
- Cheap object storage still needs lifecycle, access, and query-cost discipline.
- Log volume growth should be forecast by source and team.
- Cost controls should not erase the evidence needed for incidents.

## Practical Guidance

- Tier retention by log class when useful.
- Track noisy sources and low-value logs aggressively.
- Make deletion and archival policy clear to teams.
- Consider how retention interacts with forensic and regulatory expectations.

## Principal Review Lens

- Which log source gives least value per cost?
- Are we retaining data because it is useful or because nobody decided not to?
- What would be hardest to investigate if retention dropped tomorrow?
- Which cost lever is safest to pull first?
