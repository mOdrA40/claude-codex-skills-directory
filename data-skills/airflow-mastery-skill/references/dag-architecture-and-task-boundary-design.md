# DAG Architecture and Task Boundary Design (Airflow)

## Rules

- DAGs should represent orchestration intent clearly, not embed uncontrolled business logic.
- Task boundaries should support retries, visibility, and ownership.
- Keep DAGs readable enough that on-call can reason about them quickly.
- Avoid giant DAGs that entangle unrelated failure domains.

## Practical Guidance

- Separate ingestion, transformation, validation, and publish stages when operationally useful.
- Make external dependency boundaries explicit.
- Keep dynamic DAG generation under strong discipline.
- Design tasks for clear inputs, outputs, and retry behavior.

## Architecture Heuristics

### DAGs should reveal failure domains

Good DAG structure makes it obvious which tasks belong together operationally, which should fail or retry independently, and which boundaries matter for blast-radius control.

### Task splitting should earn its complexity

More tasks improve visibility only if they also improve recoverability, ownership clarity, and retry safety. Excessive fragmentation can make orchestration harder to understand without improving resilience.

### Dynamic generation needs stronger governance than static DAGs

Dynamic behavior can scale teams well, but it also multiplies hidden complexity in review, debugging, and support if naming, ownership, and dependency patterns are weak.

## Common Failure Modes

### DAG readability collapse

The workflow technically works, but on-call engineers cannot quickly explain the critical path, retry posture, or dependency surface under pressure.

### Task boundaries by code preference

Tasks are split according to implementation convenience rather than operational recoverability or business semantics.

### Dynamic DAG sprawl

Generated DAGs proliferate faster than platform governance, leaving operators with too many near-duplicates and weak ownership clarity.

## Principal Review Lens

- Which DAG is too large for safe support?
- Are task boundaries helping recoverability or just splitting code arbitrarily?
- What DAG pattern most threatens scheduler clarity?
- Which workflow should be decomposed or simplified first?
- Which boundary currently looks neat in code but creates the most operator confusion in incidents?
