# Drift Detection and Observability

## Principle

Drift is not only a data science concern. In production it is an operational signal that the system may be making increasingly wrong decisions while still appearing healthy at the infrastructure level.

## Signals

- feature distribution drift
- label delay and quality drift
- embedding drift
- retrieval hit quality degradation
- fallback rate changes
- confidence distribution shifts

## Review Questions

- what signals indicate the model is still accurate enough to trust?
- how quickly can operators detect silent degradation?
- what rollback or disable policy exists when drift exceeds safe bounds?
