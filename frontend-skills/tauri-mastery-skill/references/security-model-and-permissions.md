# Security Model and Permissions in Tauri

## Principle

Tauri security depends on tight command design, careful local permissions, and explicit control of what the embedded frontend can ask the native layer to do.

## Common Failure Modes

- permissions broader than necessary
- frontend assumptions treated as trusted intent
- plugin usage expanding native attack surface casually

## Review Questions

- what permission is actually required?
- what part of the frontend should not be trusted with this capability?
- how would a bad plugin or command misuse be detected?
