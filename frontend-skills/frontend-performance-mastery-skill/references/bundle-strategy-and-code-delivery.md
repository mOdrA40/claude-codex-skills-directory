# Bundle Strategy and Code Delivery

## Principle

Bundle strategy is route strategy, dependency strategy, and execution strategy combined. Smaller bundles matter only when they reduce real user wait and device pressure.

## Common Failure Modes

- one shared dependency inflating every route
- code splitting that helps build metrics but not user flow
- client-heavy architecture making bundle wins impossible later

## Review Questions

- what code is on the critical path for first useful interaction?
- which dependency should not be global?
- what route class suffers most from current bundle decisions?
