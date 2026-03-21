# Alerting Strategy and Fatigue

## Rules

- Alerts should map to customer impact, data loss risk, security risk, or imminent saturation.
- Warning and critical levels should imply different operator actions.
- Page humans for conditions that need timely human judgment, not every anomaly.
- Silence and inhibition policies should reduce noise without hiding real failures.

## Common Mistakes

- Alerting on symptoms with no action path.
- Paging on every node blip while missing service-level burn rate problems.
- Letting alert naming, labels, and ownership drift into chaos.
- Treating alert volume as observability maturity.

## Principal Review Lens

- Which alerts would the team delete tomorrow with no downside?
- Which missing alert would hurt the most in a real incident?
- Are we alerting on user impact or just internal motion?
- Can on-call map each page to a clear runbook and owner?
