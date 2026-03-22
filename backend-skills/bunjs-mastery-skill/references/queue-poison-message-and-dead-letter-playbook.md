# Queue Poison Message and Dead-Letter Playbook for Bun Services

## Rules

- classify transient vs terminal failures
- define retry caps
- expose dead-letter cause and age
- make replay safe and observable
- keep poison traffic from starving healthy traffic

## Agent Questions

- why is this message retryable?
- what signal moves it to dead-letter?
- how is duplicate side effect prevented on replay?
- can one poison event monopolize worker concurrency?
