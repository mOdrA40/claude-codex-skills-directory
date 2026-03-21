# Partitioning, Shuffle, and Data Layout

## Rules

- Shuffle is one of the main cost and latency drivers in Spark.
- Partitioning and file layout should be designed around dominant query and transform patterns.
- Data layout mistakes create recurring platform tax.
- Optimizing layout is often higher leverage than chasing executor flags.

## Practical Guidance

- Watch skew, partition explosion, small files, and wide dependencies.
- Design storage layout to reduce expensive scans and reshuffles.
- Validate partitioning assumptions under production data skew.
- Align upstream write patterns with downstream Spark economics.

## Principal Review Lens

- Which workload is paying the most shuffle tax?
- Are we blaming Spark runtime for poor data layout?
- What skew pattern will hurt worst at scale?
- Which layout change would most improve platform cost and predictability?
