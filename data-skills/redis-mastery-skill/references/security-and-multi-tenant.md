# Security and Multi-Tenant Redis

## Rules

- Shared Redis requires namespace, access, and blast-radius discipline.
- Secrets, sessions, and tokens need extra handling care.
- Tenant separation should survive debugging and incident response.
- Protect admin commands and network surfaces carefully.

## Principal Review Lens

- Which tenant can affect or infer another tenant's data?
- What command or access path is too powerful?
- How are secrets and session keys protected operationally?
