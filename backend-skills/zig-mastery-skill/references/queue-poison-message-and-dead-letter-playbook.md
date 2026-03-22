# Queue Poison Message and Dead-Letter Playbook for Zig Services

## Rules

- classify transient vs terminal failure
- cap retries explicitly
- expose dead-letter count, age, and cause
- make replay safe and observable
- keep poison traffic from monopolizing worker capacity

## Agent Questions

- why is this message retryable?
- when should it dead-letter?
- how is replay made safe for side effects?
- can poison traffic starve healthy work?
