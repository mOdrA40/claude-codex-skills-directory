# Supply-Chain Integrity and Provenance

## Rules

- Builds should preserve trustworthy provenance from source to artifact.
- Signing, attestation, and verification are only useful when workflows are consistently enforced.
- Supply-chain controls should target real risk, not just compliance checklists.
- Artifact integrity must survive retries, parallelism, and multi-stage workflows.

## Practical Guidance

- Make provenance artifacts easy to retrieve and verify.
- Separate untrusted source validation from trusted release signing paths.
- Keep build steps reproducible where feasible.
- Standardize how platform teams and service teams reason about trust.

## Principal Review Lens

- Can the team prove what source produced this artifact?
- Which build step most weakens trust today?
- Are provenance controls operationally meaningful or performative?
- What supply-chain control deserves enforcement next?
