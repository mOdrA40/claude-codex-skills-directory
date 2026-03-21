# Runner Isolation, Build Security, and Secret Handling

## Rules

- Pipeline execution environment is part of your security boundary.
- Build steps with secrets, signing, or deployment capability require stronger isolation.
- Shared execution environments should not silently widen trust.
- Secret handling should minimize exposure in logs, volumes, and task definitions.

## Practical Guidance

- Separate trusted and untrusted workloads deliberately.
- Control service account, volume, and network posture tightly.
- Watch for artifact poisoning and workspace persistence risks.
- Keep secret usage scoped to the minimum tasks that need it.

## Principal Review Lens

- Which task has more power than it should?
- Are we trading security for throughput without admitting it?
- What secret path is most likely to leak?
- Which isolation boundary most needs hardening?
