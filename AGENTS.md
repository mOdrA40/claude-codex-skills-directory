# Skill Authoring Guide (Codex Skills Repo)

This repository contains multiple “skills” (each skill is a directory with a `SKILL.md` entry point plus references/scripts). Use this file as the canonical template and quality bar for creating and maintaining skills.

## Goals

- Make skills immediately usable: clear triggers, scoped instructions, and actionable outputs.
- Keep content senior/principal-level: focus on architecture, failure modes, and production readiness.
- Provide both ❌/✅ examples and “why” (tradeoffs), not just rules.

## Repository conventions

- **Language:** all new or edited `*.md` files must be written in **English** (chat can be any language, for this cases u answer in Indonesian).
- **Links:** use **relative links** inside a skill directory, e.g. `references/tooling.md` (avoid hard-coded `skills/...` paths).
- **Structure:** prefer `references/` for long docs and `scripts/` for runnable automation.

Recommended layout:

```
<skill-name>/
  SKILL.md
  references/
    *.md
  scripts/
    *.ps1 / *.sh / *.py
```

## `SKILL.md` requirements (entry point)

Every `SKILL.md` must:

1. Include YAML frontmatter:
   - `name`: stable, unique identifier
   - `description`: when to use the skill + what it covers
2. Start with “Operate” / “How to use” guidance (what the assistant does first).
3. Provide a minimal “default standard” section (the baseline opinionated rules).
4. Provide a “Validation commands” section (format/test/lint/security checks).
5. Link to deeper topics under `references/` using relative links.
6. Include at least one small ❌/✅ example that reflects real production pitfalls.

Minimal template:

```md
---
name: <skill-id>
description: |
  <what it is>
  Use when: <use cases>
---

# <Skill Title>

## Operate
- Confirm goal/scope/constraints/NFRs/done
- Prefer small diffs + tests
- Explain tradeoffs briefly; output actionable steps

## Default Standards
- <rules>

## Validation Commands
- <commands>

## References
- [Topic](references/topic.md)
```

## Content quality bar (Senior → Principal)

Include these topics when relevant:

- Architecture boundaries (transport/use-case/domain/adapters) and dependency direction.
- Error taxonomy and mapping across boundaries (HTTP/queue).
- Timeouts, retries, backoff/jitter, idempotency, backpressure.
- Concurrency ownership (every goroutine has an owner + stop condition).
- Observability (logs/metrics/traces) with incident response in mind.
- Security posture (input validation, SSRF, secrets handling, least privilege).
- Operational ergonomics (graceful shutdown, health/readiness, migrations).

Prefer “guardrails” (safe defaults) over abstract theory.

## Examples policy

- For every “rule”, include at least one of:
  - ❌/✅ code snippet
  - “failure story” (what breaks in production)
  - decision matrix (“choose A vs B when…”)
- Prefer standard library patterns unless there’s a clear win from a dependency.

## Scripts policy

- Scripts should be safe-by-default and optional:
  - `go_sanity.*`: format/vet/test/race/lint if present
  - `go_security.*`: `govulncheck` + optional `gosec`/`staticcheck` if present
- Scripts must be idempotent and print clear section headers.
- Never assume global tools exist; detect and skip with a message.

## Review checklist (for skill PRs)

- No broken links (all `references/...` targets exist).
- No hard-coded paths like `skills/<name>/...`.
- Entry point is short; deep docs moved to `references/`.
- Content includes ❌/✅ examples and explicit tradeoffs.
- Validation commands are correct for the ecosystem.
