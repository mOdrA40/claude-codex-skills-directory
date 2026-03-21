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

## Principal Review Lens

- Which DAG is too large for safe support?
- Are task boundaries helping recoverability or just splitting code arbitrarily?
- What DAG pattern most threatens scheduler clarity?
- Which workflow should be decomposed or simplified first?
