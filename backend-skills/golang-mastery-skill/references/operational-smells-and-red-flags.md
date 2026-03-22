# Operational Smells and Red Flags in Go Services

## Red Flags

- goroutine fan-out with no bound or owner
- context ignored on outbound IO
- retries layered over already saturated dependencies
- queue consumers with no lag or age visibility
- autoscaling used without understanding the real bottleneck
- rollout-sensitive schema change with no compatibility plan

## Agent Review Questions

- where does pressure accumulate first?
- can one tenant or workload shape monopolize shared resources?
- do retries amplify rather than heal?
- what signal tells operators to stop rollout?

## Principal Heuristics

- If goroutine growth is the only scaling answer, architecture is likely hiding pain.
- If queue age is invisible, async correctness is unprovable during incidents.
- If rollback is unclear, deployment safety is weak.
