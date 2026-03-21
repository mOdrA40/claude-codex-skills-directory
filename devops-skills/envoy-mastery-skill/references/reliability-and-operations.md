# Reliability and Operations (Envoy)

## Operational Defaults

- Monitor config push status, proxy health, connection churn, retry volume, and upstream error behavior.
- Keep config rollouts staged and reversible.
- Separate shared proxy platform incidents from tenant-specific config incidents.
- Document safe downgrade and emergency traffic isolation workflows.

## Run-the-System Thinking

- Envoy fleets at edge or sidecar scale deserve explicit SLOs.
- Capacity planning should include connection behavior, TLS cost, and traffic burstiness.
- On-call should know which knobs are safe during overload.
- Operational trust comes from boring config and good rollout discipline.

## Principal Review Lens

- Which Envoy failure mode has the biggest blast radius?
- Can the team rollback config safely under pressure?
- What operational practice most improves confidence in the proxy layer?
- Are we operating a robust data plane or a config experiment?
