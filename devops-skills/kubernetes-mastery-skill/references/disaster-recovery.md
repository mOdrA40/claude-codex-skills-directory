# Disaster Recovery

## Rules

- Separate cluster recovery from workload recovery and data recovery.
- Backups must include the control plane assumptions that matter.
- Restore drills should test real service startup, not just object existence.
- Define what is rebuilt versus restored.

## Principal Review Lens

- How long does full platform recovery really take?
- Which workload is hardest to recover safely?
- Are recovery procedures dependent on tribal knowledge?
