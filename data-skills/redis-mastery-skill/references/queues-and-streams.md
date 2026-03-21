# Queues and Streams (Redis)

## Rules

- Queues in Redis are fine when semantics and durability limits are explicit.
- Streams add structure but still require consumer ownership and replay thinking.
- Poison message and duplicate handling must be designed.
- Do not let convenience hide delivery guarantees.

## Messaging Heuristics

### Use Redis messaging when simplicity truly wins

Redis queues and streams are strongest when the workload values simplicity, low operational overhead, and bounded semantics more than heavyweight durability or large-scale replay guarantees.

### Delivery semantics must be explained in business terms

Teams should know whether the system tolerates:

- duplicates
- lost work on failure
- replay after consumer outage
- backlog growth during dependent-system slowdown

### Consumer operability matters as much as producer convenience

It is easy to enqueue work. The harder question is whether operators can reason about lag, stuck consumers, replay, poison messages, and retention pressure under real incidents.

## Common Failure Modes

### Queue convenience drift

Redis starts as a simple work queue, then gradually inherits expectations that belong to a more explicit messaging platform.

### Replay ambiguity

Streams support replay, but nobody defines when it is safe, how much load it creates, or what duplicates mean for downstream systems.

### Hidden backlog risk

The queue appears healthy until one consumer slowdown or dependency outage turns retained work into a memory and recovery problem.

## Principal Review Lens

- Why Redis queue/stream over Kafka or RabbitMQ here?
- What failure loses or duplicates work?
- Who owns replay, retention, and dead-letter behavior?
- What messaging promise are we implicitly making that Redis may not safely keep at higher scale?
