# Incident Runbooks (Kubernetes)

## Rules

- Cover bad rollouts, node pressure, DNS/network failures, control plane symptoms, and autoscaling pathologies.
- Stabilize critical services before perfect diagnosis.
- Include safe rollback and blast-radius reduction steps.
- Recovery must be verified from both platform and user perspectives.

## Principal Review Lens

- Can on-call identify cluster issue versus app issue in minutes?
- Which emergency action risks wider outage?
- What confirms real recovery rather than churn?
