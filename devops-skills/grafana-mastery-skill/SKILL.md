---
name: grafana-principal-engineer
description: |
  Principal/Senior-level Grafana playbook for dashboards, alerting UX, multi-source observability, access design, operational governance, and decision-support visualization.
  Use when: designing dashboards, reviewing observability UX, building operational workflows, unifying metrics/logs/traces, or governing Grafana in production.
---

# Grafana Mastery (Senior → Principal)

## Operate

- Start from operator workflows, not aesthetic dashboards.
- Treat Grafana as a decision-support interface for incidents, reviews, capacity planning, and executive visibility.
- Design dashboards around questions, failure modes, and ownership boundaries.
- Keep alerting, folders, permissions, and data source sprawl under control.

## Default Standards

- Dashboards should support triage in minutes.
- Variables and templating must reduce toil, not hide logic.
- Folder, team, and permission models should reflect ownership.
- Logs, metrics, and traces should converge on the same service reality.
- Provisioning and change review should be predictable.

## References

- Dashboard design and operator workflows: [references/dashboard-design-and-operator-workflows.md](references/dashboard-design-and-operator-workflows.md)
- Panels, queries, and performance: [references/panels-queries-and-performance.md](references/panels-queries-and-performance.md)
- Alerting UX and escalation: [references/alerting-ux-and-escalation.md](references/alerting-ux-and-escalation.md)
- Data sources and correlation: [references/data-sources-and-correlation.md](references/data-sources-and-correlation.md)
- Access control and multi-team governance: [references/access-control-and-multi-team-governance.md](references/access-control-and-multi-team-governance.md)
- Provisioning and dashboard as code: [references/provisioning-and-dashboard-as-code.md](references/provisioning-and-dashboard-as-code.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
