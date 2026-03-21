# Multi-Tenant Boundaries and Rate Limits in Bun APIs

## Principle

Rate limiting is not just anti-abuse. In multi-tenant systems it is part of fairness, blast-radius control, and cost containment.

## Rules

- identify tenant or caller at the edge
- separate user-level and tenant-level limits
- avoid global limits that let one tenant starve others
- propagate tenant identity into logs and metrics safely
- define degraded behavior for over-budget tenants

## Review Questions

- what happens when one tenant bursts 100x normal traffic?
- are cache and queue keys tenant-aware?
- can optional features be shed before core API paths fail?
