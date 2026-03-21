# Operations and Upgrades

## Rules

- Upgrades should be staged and reversible.
- Node maintenance must be validated against queue type and HA behavior.
- Broker operations need clear ownership and maintenance windows.
- Capacity and recovery planning should influence upgrade timing.

## Principal Review Lens

- What queue type behavior changes during maintenance?
- Can operators predict recovery time during rolling changes?
- Which upgrade step carries the largest blast radius?
