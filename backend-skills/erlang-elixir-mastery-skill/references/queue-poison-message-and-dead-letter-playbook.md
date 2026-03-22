# Queue Poison Message and Dead-Letter Playbook on the BEAM

## Rules

- classify transient vs terminal failure
- cap retries explicitly
- expose dead-letter volume, age, and cause
- make replay safe and observable
- prevent poison traffic from monopolizing consumers or partitions

## Agent Questions

- what makes this message retryable?
- when should it dead-letter?
- how can replay avoid duplicate side effects?
- can poison traffic block healthy work?
