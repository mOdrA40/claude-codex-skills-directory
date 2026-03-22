# Queue Poison Message and Dead-Letter Playbook for Go Services

## Rules

- classify transient vs terminal failures
- bound retries explicitly
- expose dead-letter volume, age, and cause
- make replay safe for idempotent workflows
- stop poison messages from monopolizing partitions or workers

## Agent Questions

- what proves this message is retryable?
- when should it be dead-lettered?
- how do operators replay without duplicating writes?
- can poison traffic block healthy work?
