# Sampling Strategy and Critical Workflows

## Rules

- Sampling policy should preserve traces for the workflows that matter most.
- Error paths, high-latency requests, and business-critical journeys deserve explicit consideration.
- Sampling should be visible to service owners and responders.
- Cost control should not blind the system during major incidents.

## Practical Guidance

- Review how sampling affects trace correlation and search usefulness.
- Different services may justify different strategies.
- Keep head-versus-adaptive sampling implications understandable.
- Revisit policy when traffic or business criticality changes.

## Principal Review Lens

- Which important workflow is most under-sampled today?
- Are we saving cost intelligently or hiding the hard problems?
- What sampling bias is distorting architecture decisions?
- Which service should have a different policy right now?
