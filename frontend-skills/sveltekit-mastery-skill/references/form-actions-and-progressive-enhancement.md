# Form Actions and Progressive Enhancement

## Principle

SvelteKit form actions are powerful because they align with the web platform. Teams should resist rebuilding heavy client-side mutation systems unless the product really needs them.

## Action Design Model

Form actions should answer clearly:

- what data is validated where
- what mutation boundary the action represents
- what success state returns to the user
- what failure remains local vs route-level
- what no-JS behavior is promised as a baseline

## Use Actions Well

- keep validation and mutation ownership explicit
- model success, retry, and failure states intentionally
- use progressive enhancement where it improves UX without hiding complexity

## Progressive Enhancement Heuristics

### Start from trustworthy baseline behavior

If the form works meaningfully without enhancement, then enhancement can improve speed and smoothness without becoming a hidden dependency.

### Keep pending and retry states honest

Enhanced UX should not make the user believe a mutation is complete when the durable result is still uncertain.

### Avoid action boundaries that become mini workflow engines

If one action starts coordinating too many unrelated business steps, the route contract becomes hard to reason about and recover from.

## Common Failure Modes

- treating enhanced forms like magic and losing clear mutation ownership
- pushing too much product complexity into one action boundary
- forgetting that degraded no-JS or partial-JS behavior is still part of the system contract

### Mutation ambiguity

Users and developers cannot tell whether a failure belongs to validation, business rules, transport, or enhancement behavior.

### Enhancement-only correctness

The product appears correct with JavaScript fully enabled but has weak or broken fallback behavior under degraded conditions.

## Review Questions

- what does the form do without enhancement?
- which failures should remain on-page vs navigate away?
- when would a more explicit client mutation model be justified?
- what mutation flow is currently too magical to debug confidently?
