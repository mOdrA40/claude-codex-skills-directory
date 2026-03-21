# Failure States and Resilience in Compose/KMP Apps

## Rules

- distinguish loading, empty, stale, failed, and offline states
- model retry and recovery intentionally
- do not let shared abstractions hide platform-specific degraded behavior
