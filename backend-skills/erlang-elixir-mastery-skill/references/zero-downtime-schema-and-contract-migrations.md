# Zero-Downtime Schema and Contract Migrations on the BEAM

## Core Principle

BEAM uptime does not make deploys automatically safe. Compatibility across nodes, workers, events, and schemas still needs deliberate rollout design.

## Safe Rules

- prefer additive schema and contract changes first
- allow old and new nodes to coexist during rollout
- treat Broadway consumers, Phoenix endpoints, and background jobs as separate compatibility surfaces
- define rollback before changing producer or consumer assumptions
- delay cleanup until mixed-version risk is gone

## Agent Checklist

- can old and new nodes process the same workflow safely?
- what happens to queued messages during rollout?
- do consumers understand both event or payload versions?
- does rollback preserve correctness and operability?
- where would compatibility failure show up first: mailboxes, queue lag, or request errors?
