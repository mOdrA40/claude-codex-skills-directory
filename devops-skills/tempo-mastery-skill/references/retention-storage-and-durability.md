# Retention, Storage, and Durability

## Rules

- Retention should be tied to debugging value and compliance need explicitly.
- Trace durability expectations must be honest about sampling, ingest drops, and backend dependencies.
- Storage cost controls should not erase critical forensic windows unknowingly.
- Object storage lifecycle and access policy are part of trace platform design.

## Practical Guidance

- Define different retention classes where useful.
- Track the cost of longer retention against real usage.
- Test restore and access patterns if trace history matters operationally.
- Communicate clearly what traces may be missing by design.

## Principal Review Lens

- What retention window actually matters to responders?
- Are we storing traces longer than their real value?
- Which durability promise is implied but not actually true?
- What storage decision most affects future platform trust?
