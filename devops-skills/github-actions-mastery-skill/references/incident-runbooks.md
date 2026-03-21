# Incident Runbooks (GitHub Actions)

## Cover at Minimum

- Runner outage or starvation.
- Secret or token access failure.
- Bad reusable workflow rollout.
- Release pipeline failure.
- Supply-chain or action trust incident.
- Platform-wide flaky execution or event trigger issue.

## Response Rules

- Restore safe release capability before optimizing broad CI convenience.
- Prefer targeted rollback of shared automation over wide panic edits.
- Preserve workflow logs, artifacts, and environment gate evidence.
- Communicate clearly whether failure is GitHub-hosted, self-hosted, or workflow logic.

## Principal Review Lens

- Can responders restore safe deploy capability quickly?
- Which emergency action most risks making trust worse?
- What proves the platform is healthy again?
- Are runbooks realistic for shared CI/CD outages?
