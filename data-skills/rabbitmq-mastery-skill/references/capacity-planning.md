# Capacity Planning

## Rules

- Plan for queue depth, burst publish rate, consumer throughput, and failure headroom.
- Memory, disk alarms, and quorum overhead all matter.
- Benchmarks should model real message size and acknowledgement behavior.
- HA topology changes capacity math materially.

## Principal Review Lens

- What fails first under burst traffic?
- How much headroom remains during node loss?
- Which queue or tenant dominates platform cost?
