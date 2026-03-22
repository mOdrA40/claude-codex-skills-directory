# High-Risk Change Preflight Checklist on the BEAM

## Use This Before

- schema or contract changes
- OTP structure changes
- retry policy changes
- consumer semantics changes
- dependency upgrades with production impact

## Checklist

- define rollback trigger
- confirm mixed-version compatibility
- confirm observability for new failure modes
- confirm blast radius and containment plan
- confirm mailbox, queue, and dependency impact is understood
