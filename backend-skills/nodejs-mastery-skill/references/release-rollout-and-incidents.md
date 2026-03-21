# Release, Rollout, and Incident Handling in Node.js Services

## Goal

A backend skill is incomplete if it explains how to write code but not how to ship and operate it safely.

## Rollout Basics

Every service should define:

- readiness behavior during deploys
- compatibility expectations for config and schema changes
- rollback trigger conditions
- log/metric fields identifying release version
- what happens to background consumers during rollout

## Common Deploy Failure Modes

- readiness passes before dependencies are truly usable
- old and new versions disagree on payload or event schema
- consumers double-process work during rolling restarts
- hidden startup migrations block the process too long

## Incident Questions

- Is the issue release-specific or load-specific?
- Which version first showed the regression?
- Can the service degrade safely without full rollback?
- Are retries and queue consumers amplifying the incident?

## Review Checklist

- version is observable
- readiness is meaningful
- shutdown behavior is tested
- rollback trigger is explicit
- background work behavior during rollout is defined
