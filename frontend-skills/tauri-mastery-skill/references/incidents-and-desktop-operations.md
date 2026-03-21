# Incidents and Desktop Operations in Tauri

## Principle

Desktop incidents may involve updater behavior, local filesystem state, version skew, command misuse, or packaging regressions. Teams need local-operational visibility, not just web-style logs.

## Operational Realities

Desktop applications fail differently from web apps because:

- users may stay on old versions longer
- local state can become corrupted or semantically outdated
- OS-specific behavior matters materially
- recovery paths often depend on filesystem, config, or updater behavior

This means desktop operations must model version cohorts, local-state damage, and OS-specific blast radius more explicitly than browser-only systems.

## Incident Classes

- bad update or partial upgrade behavior
- corrupted or incompatible local state
- command failures tied to OS context or permissions
- desktop-only performance regressions

## Triage Heuristics

### Isolate by version and OS first

Before chasing deep root cause, determine:

- which app versions are affected
- which operating systems or OS versions are involved
- whether the issue appears only after update, only on clean installs, or only on long-lived user state

### Distinguish local-state incidents from binary incidents

Many Tauri issues look like “the app broke after release” but the important distinction is whether:

- the binary is wrong
- updater behavior is wrong
- persisted state is incompatible
- a command or permission assumption broke under one environment

### Prefer recovery-first thinking

If the root cause is not immediately obvious, teams should still know whether they can:

- disable a feature path
- preserve user data while downgrading behavior
- repair or reset local state safely
- stop updater rollout for affected cohorts

## Recovery Playbook Heuristics

### Preserve user data when possible

Desktop incidents can damage trust badly if the first response is to tell users to delete all local state without understanding what can be saved.

### Distinguish support actions from engineering actions

Support may need a simple recovery sequence while engineering needs deeper cohort and version analysis.

### Document safe reset boundaries

Teams should know which local directories, caches, configs, or migrations can be reset independently and which actions are destructive.

## Common Failure Modes

### Version-skew blindness

The team reasons as if all users upgraded immediately, so incidents tied to older cohorts remain invisible or misclassified.

### Local-state damage without operator playbook

The app can enter a bad state locally, but support and engineering do not share a clear recovery model.

### Desktop-only regressions hidden by web-style logging

The team has logs, but they lack the OS, version, command, and local-environment context needed to classify failures correctly.

### Recovery by folklore

The only known fixes live in the heads of one or two engineers or support staff instead of an explicit operator playbook.

## Review Questions

- what changed in app version, updater behavior, or local state assumptions?
- can the issue be isolated to one OS or version cohort?
- what rollback or recovery path exists for local-state damage?
- what desktop incident would currently take longest to distinguish from generic app failure?
- what recovery step is safe for support to recommend today, and what step is too destructive?
