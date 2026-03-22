# Tenant Fairness and Noisy Neighbor Playbook on the BEAM

## Controls

- tenant-aware throttling and quotas
- per-tenant or per-workload caps where justified
- tenant-level visibility into mailbox growth, queue age, and failure rate
- selective degradation of low-value traffic first

## Incident Questions

- which tenant or workload shape dominates mailbox pressure?
- where is tenant identity lost across messages or consumers?
- what gets throttled first to preserve shared health?

## Principal Heuristics

- Fairness is part of blast-radius control.
- If one tenant can monopolize shared processes, isolation is weak.
