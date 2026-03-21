# Operations and Upgrades

## Rules

- Upgrades should be staged and reversible.
- Node maintenance must be validated against queue type and HA behavior.
- Broker operations need clear ownership and maintenance windows.
- Capacity and recovery planning should influence upgrade timing.

## Upgrade Heuristics

### Upgrades are topology events, not package events

Changing RabbitMQ versions or operating brokers during maintenance affects queue availability, mirror/quorum behavior, backlog movement, and how consumers experience the platform.

### Queue type should shape upgrade choreography

Classic, quorum, and stream-based deployments do not create identical maintenance and recovery behavior. Upgrade plans should reflect those semantic and operational differences.

### Reversibility must remain real

An upgrade is only operationally safe if the team can describe what rollback means, what data or client assumptions are at risk, and how long partial recovery really takes.

## Common Failure Modes

### Maintenance by generic runbook

Teams apply a standard broker procedure without adapting to current queue mix, backlog shape, or consumer criticality.

### Rolling-change optimism

A rolling sequence is assumed safe, but in practice queue behavior, client reconnection, or backlog movement creates more pain than planned.

### Ownership gap during change windows

Application, platform, and on-call teams do not share one clear view of who decides rollback, throttle, or isolation during upgrade trouble.

## Principal Review Lens

- What queue type behavior changes during maintenance?
- Can operators predict recovery time during rolling changes?
- Which upgrade step carries the largest blast radius?
- Which maintenance assumption is least likely to hold under real backlog and client behavior?
