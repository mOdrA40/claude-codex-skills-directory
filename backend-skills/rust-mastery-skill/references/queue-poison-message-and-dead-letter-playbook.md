# Queue Poison Message and Dead-Letter Playbook for Rust Services

## Rules

- classify transient vs terminal failure
- bound retries explicitly
- expose dead-letter count, age, and cause
- ensure replay is safe and observable
- prevent poison traffic from monopolizing workers or partitions

## Agent Questions

- what makes this message retryable?
- when should it dead-letter?
- how can replay avoid duplicate side effects?
- can one poison message block healthy work?
