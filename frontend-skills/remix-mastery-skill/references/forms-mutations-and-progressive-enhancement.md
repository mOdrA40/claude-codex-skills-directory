# Forms, Mutations, and Progressive Enhancement in Remix

## Principle

Remix mutation flow should preserve web fundamentals while still supporting modern UX. The challenge is not whether enhancement exists, but whether mutation ownership remains obvious.

## Common Failure Modes

- form flows that work only under ideal JS behavior
- mutation states not modeled clearly for users
- retry and validation logic split across too many layers

## Review Questions

- what is the no-JS or degraded behavior?
- what does the user see during pending and failed mutation states?
- which part of this flow is too magical?
