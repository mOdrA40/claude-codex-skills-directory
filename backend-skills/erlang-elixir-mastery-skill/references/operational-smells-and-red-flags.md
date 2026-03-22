# Operational Smells and Red Flags on the BEAM

## Red Flags

- one GenServer owning unrelated responsibilities
- mailbox growth with no explicit overload posture
- unbounded async fan-out or consumer concurrency
- deploy-sensitive changes with no mixed-version thinking
- retries amplifying backlog and restart storms
- one tenant or workload shape dominating shared processes

## Agent Review Questions

- which process family absorbs the pressure first?
- does this change widen subtree blast radius?
- can operators distinguish dependency slowness from mailbox collapse?
- what signal should halt rollout or topology change?

## Principal Heuristics

- OTP structure without operability is architecture theater.
- If mailbox growth is invisible, backpressure is unmanaged.
- If restart intensity is the first signal, detection is too late.
