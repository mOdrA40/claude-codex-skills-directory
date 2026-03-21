# Observability and Debugging (cert-manager)

## Rules

- Observability should reveal issuance, renewal, challenge failures, and secret update behavior clearly.
- Debugging must distinguish issuer problems, DNS/HTTP challenge problems, and workload secret consumption issues quickly.
- Dashboards should prioritize expiry risk and failed automation paths.
- Certificate operations should be visible enough to prevent surprise expiry incidents.

## Useful Signals

- Renewal status, challenge failures, issuer health, secret update timing, and expiry horizons.
- Correlate with ingress, DNS, and workload rollout signals.
- Preserve logs and events for failed issuance and renewal.
- Standardize visibility for critical certificate inventories.

## Principal Review Lens

- Can operators identify expiring critical certs early enough?
- Which missing signal most slows cert-manager debugging today?
- Are teams blaming cert-manager for DNS or ingress problems too often?
- What observability change most reduces expiry risk?
