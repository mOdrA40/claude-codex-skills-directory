# Replication and Elections (MongoDB)

## Rules

- Elections are normal; application behavior during them must be known.
- Monitor lag, failover duration, and write concern impact.
- Read preference should match staleness tolerance explicitly.
- Test client reconnect and retry behavior during stepdown.

## Principal Review Lens

- What user-visible errors appear during primary changes?
- Is replica lag safe for read traffic here?
- Which services silently assume always-primary availability?
