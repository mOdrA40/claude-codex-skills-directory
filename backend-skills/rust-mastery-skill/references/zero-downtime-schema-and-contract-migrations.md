# Zero-Downtime Schema and Contract Migrations in Rust Services

## Core Principle

Rust correctness at compile time does not remove rollout risk. Old and new versions must coexist safely during deploys.

## Safe Rules

- prefer expand-and-contract over synchronized switching
- make compatibility explicit for HTTP, events, queues, and DB schema
- keep rollback possible without manual heroics
- validate background consumers separately from request-serving path
- treat derived caches and feature flags as compatibility surfaces too

## Agent Checklist

- can old and new versions coexist?
- do consumers understand both event forms?
- can rollback happen without data repair?
- what part of the system would double-process during rollout?
- is readiness truly traffic-safe?
