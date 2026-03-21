# Repair, Anti-Entropy, and Operational Safety

## Rules

- Repair is part of the data-correctness model and operational design, not optional maintenance theater.
- Anti-entropy workflows need scheduling, visibility, and ownership.
- Node maintenance and topology changes should account for repair posture.
- Operational shortcuts around repair can create future correctness pain.

## Practical Guidance

- Make repair cadence explicit and realistic.
- Watch streaming, hinted handoff, and repair side effects during peak load.
- Test node replacement and decommission workflows with actual procedures.
- Keep repair strategy aligned with replication and failure assumptions.

## Principal Review Lens

- Can the team explain current repair health without guessing?
- Which maintenance event most threatens correctness today?
- Are we under-repairing because the work is expensive?
- What practice would most improve long-term cluster trust?
