# Stack Design and Ownership Boundaries (Pulumi)

## Rules

- Stack boundaries should reflect ownership, lifecycle, and recovery domains.
- One giant stack is rarely a maturity signal.
- Shared stacks require stronger review and operational clarity.
- Stack references should make dependencies visible, not magical.

## Design Guidance

- Separate foundation, shared platform, and application concerns where blast radius differs.
- Make environment strategy explicit rather than implicit in code branching.
- Keep stack purpose obvious to operators and reviewers.
- Avoid stack designs that require deep code knowledge to understand infra impact.

## Principal Review Lens

- Which stack is too broad for safe ownership?
- What dependency between stacks is currently too implicit?
- Can the team recover one stack without disturbing others?
- What split or consolidation would improve operability most?
