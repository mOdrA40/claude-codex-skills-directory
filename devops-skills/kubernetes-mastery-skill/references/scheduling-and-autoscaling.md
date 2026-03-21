# Scheduling and Autoscaling

## Rules

- Scheduling constraints should reflect failure domains and cost intent.
- Autoscaling requires trustworthy signals.
- HPA is not a fix for bad resource modeling.
- Pod disruption and surge behavior should be considered together.

## Principal Review Lens

- What signal is driving scale decisions and can it lie?
- Are affinity rules helping safety or trapping capacity?
- How does autoscaling behave during dependency slowdown?
