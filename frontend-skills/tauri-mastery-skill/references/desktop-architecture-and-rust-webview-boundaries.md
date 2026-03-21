# Desktop Architecture and Rust-Webview Boundaries

## Principle

Tauri architecture is strongest when frontend and Rust responsibilities are explicit. Confused boundaries create security risk and debugging pain.

## Separate Clearly

- UI rendering and interaction
- Rust commands and privileged actions
- local filesystem and OS integration
- state persistence
- update and packaging concerns

## Common Failure Modes

- command surfaces expanding without governance
- business logic split arbitrarily across Rust and frontend
- privileged local actions invoked from weakly validated UI pathways

## Review Questions

- what belongs in Rust because of privilege or control?
- what belongs in the frontend because of UX ownership?
- where is the command boundary too permissive?
