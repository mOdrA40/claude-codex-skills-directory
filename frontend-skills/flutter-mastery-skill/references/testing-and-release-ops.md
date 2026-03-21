# Testing and Release Operations in Flutter

## Principle

Flutter delivery quality depends on two things that teams often separate incorrectly:

- testing discipline that proves critical behavior
- release discipline that prevents configuration, crash, or migration regressions from reaching users unnoticed

Good teams treat them as one operating model, not two unrelated checklists.

## Testing Pyramid

### Unit tests

Use unit tests for:

- pure domain logic
- formatting and mapping rules
- validation rules
- repository policies that can be isolated
- retry/backoff calculations

These should be the cheapest and most numerous tests.

### Widget tests

Use widget tests for:

- view-state rendering logic
- loading, empty, error, and retry states
- user interactions that should not require a full device/emulator run
- navigation decisions at feature level

### Integration or end-to-end tests

Use these only for flows where the risk justifies the cost:

- authentication
- onboarding
- payments
- offline sync recovery
- deep links
- release-critical core journeys

## Golden Tests

Golden tests are useful when:

- the UI is stable enough to snapshot meaningfully
- a design system or core reusable component needs regression protection
- the visual state matrix is small and intentional

Do not use them as a substitute for behavioral tests.

## Rules

- test domain logic separately from widget composition
- use golden/snapshot tests intentionally, not mechanically
- keep flaky async timing out of tests whenever possible
- prefer deterministic test data builders over giant fixtures
- make release channel, crash visibility, and rollback discipline explicit
- treat schema migrations and local persistence compatibility as release-critical behavior

## Release Discipline

Before release, define:

- app version and build number strategy
- environment separation for dev, staging, and production
- crash and analytics visibility by release version
- remote config or feature flag fallback behavior
- compatibility expectations for local persistence and backend contracts

## Common Release Failures

### Works on debug, fails on release

Often caused by:

- environment mismatch
- minification or obfuscation assumptions
- asset loading differences
- incorrect production API configuration

### Silent data issues after update

Often caused by:

- local schema migration mistakes
- stale cache assumptions
- incompatible API payload changes

### Crash spikes after rollout

Usually made worse when teams cannot answer quickly:

- which version is crashing
- which route or screen is dominant
- whether the problem is device-specific or global

## Principal Review Questions

- what behavior is so critical that it must be protected before every release?
- which tests prove recovery behavior, not just happy paths?
- can operators correlate crash reports to exact release and feature exposure?
- what is the rollback or hotfix path if the release damages persistence or navigation?
