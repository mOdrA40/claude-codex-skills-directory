# Service Discovery and Scrape Architecture

## Rules

- Discovery strategy should reflect how workloads appear, move, and die in the platform.
- Scrape jobs should be organized by ownership, environment, and operational purpose.
- Relabeling rules must remain understandable by humans.
- Scrape timeout, interval, and target availability must match system reality.

## Common Failure Modes

- Target churn causing silent cardinality spikes.
- Misconfigured relabeling dropping critical targets or exploding label space.
- One scrape config trying to solve every environment at once.
- Blackbox and whitebox monitoring overlapping without a clear role.

## Principal Review Lens

- How quickly can an operator prove a target is being scraped correctly?
- Which relabel rule is hardest to reason about under pressure?
- Are scrape settings matched to target cost and freshness needs?
- What discovery assumption breaks first in ephemeral environments?
