# Zero-Downtime Schema and Contract Migrations in Go Services

## Core Principle

Go services should assume traffic from old and new binaries overlaps during rollout.

## Safe Rules

- prefer expand-and-contract schema changes
- keep readers tolerant before tightening writers
- separate backfill from user-facing deploy when possible
- treat events, gRPC, and HTTP contracts as compatibility surfaces
- define rollback behavior before rollout starts

## Agent Checklist

- can old and new binaries read the same data safely?
- can consumers process both event versions?
- is there a migration lock or backlog risk?
- does rollback preserve correctness?
- are idempotent replays safe during mixed-version traffic?
