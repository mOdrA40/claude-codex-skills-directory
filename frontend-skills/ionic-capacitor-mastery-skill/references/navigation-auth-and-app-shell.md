# Navigation, Auth, and App Shell Behavior

## Principle

Hybrid mobile shells amplify navigation and auth mistakes because resume behavior, deep links, and session restoration interact in messy ways.

## Common Failure Modes

- auth resolution and navigation redirects fighting each other
- deep links landing before required state is restored
- tab or shell state hiding stale session assumptions

## Review Questions

- what is the app-shell bootstrap sequence?
- how does the app recover from expired session on resume?
- where is route or tab ownership most ambiguous?
