---
name: docker-principal-engineer
description: |
  Principal/Senior-level Docker playbook for image design, build performance, runtime security, networking, debugging, CI/CD delivery, and production container operations.
  Use when: containerizing services, reviewing Dockerfiles, optimizing image supply chain, debugging container incidents, or designing container-based delivery workflows.
---

# Docker Mastery (Senior → Principal)

## Operate

- Start from workload shape, runtime constraints, security requirements, and deployment target.
- Treat Docker as software supply chain and runtime isolation, not just packaging.
- Prefer reproducible, minimal, reviewable images over clever build tricks.
- Optimize for operability: startup, health, logs, signals, and debug path.

## Default Standards

- Use multi-stage builds.
- Pin base images intentionally.
- Run as non-root where possible.
- Minimize image contents and attack surface.
- Make entrypoints explicit and signal-safe.

## References

- Image design: [references/image-design.md](references/image-design.md)
- Dockerfiles and build patterns: [references/dockerfiles-and-build-patterns.md](references/dockerfiles-and-build-patterns.md)
- BuildKit and caching: [references/buildkit-and-caching.md](references/buildkit-and-caching.md)
- Security hardening: [references/security-hardening.md](references/security-hardening.md)
- Runtime operations: [references/runtime-operations.md](references/runtime-operations.md)
- Networking: [references/networking.md](references/networking.md)
- Storage and volumes: [references/storage-and-volumes.md](references/storage-and-volumes.md)
- Compose for local environments: [references/compose-and-local-dev.md](references/compose-and-local-dev.md)
- CI/CD and supply chain: [references/ci-cd-and-supply-chain.md](references/ci-cd-and-supply-chain.md)
- Debugging containers: [references/debugging-containers.md](references/debugging-containers.md)
- Performance and resource limits: [references/performance-and-resource-limits.md](references/performance-and-resource-limits.md)
- Base image strategy: [references/base-image-strategy.md](references/base-image-strategy.md)
- Multi-arch delivery: [references/multi-arch-and-distribution.md](references/multi-arch-and-distribution.md)
- Observability: [references/observability.md](references/observability.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
