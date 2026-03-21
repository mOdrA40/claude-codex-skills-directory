# Multi-Arch and Distribution

## Rules

- Multi-arch support should follow real deployment needs.
- Build, test, and release paths must validate each target architecture.
- Registry distribution strategy affects latency and reliability.
- Manifest complexity should be justified by platform reality.

## Principal Review Lens

- Which architecture is actually required in production?
- Are we testing all published images or only hoping?
- Where can distribution lag or inconsistency hurt releases?
