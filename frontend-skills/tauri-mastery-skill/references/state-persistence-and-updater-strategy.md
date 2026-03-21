# State, Persistence, and Updater Strategy in Tauri

## Principle

Desktop apps must treat persistence and updater behavior as architecture, not packaging detail. A bad update can break local state in ways web apps do not face.

## Persistence Model

Teams should define clearly:

- what data is ephemeral
- what data survives restart
- what data survives update
- what data must be migrated carefully
- what data should be recoverable or resettable

Desktop users often keep applications for long periods and skip versions, so compatibility assumptions matter more than in continuously refreshed web surfaces.

## Common Failure Modes

- local state changes without migration discipline
- updater behavior that cannot be correlated with incidents
- assumptions that users are always on the latest version

### Version skew denial

The team designs as if all users update immediately, even though real desktop cohorts may lag for weeks or months.

### Local-state fragility

The product becomes dependent on state that is hard to repair, migrate, or explain after a bad update.

## Updater Heuristics

### Treat updates as compatibility events

An update is not just a binary replacement. It can change:

- command behavior
- filesystem expectations
- config shape
- local data interpretation

### Preserve recovery paths

Users and operators need a believable path when an update damages local state or environment assumptions.

## Review Questions

- what data survives update and uninstall?
- how are migrations handled?
- what compatibility window must the app support in the real world?
- what failure after update would be hardest for users to recover from today?
