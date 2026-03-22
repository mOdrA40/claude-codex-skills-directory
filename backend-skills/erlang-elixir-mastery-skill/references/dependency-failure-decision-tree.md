# Dependency Failure Decision Tree on the BEAM

## Database or External Service Slow

- protect mailboxes and queue age first
- degrade optional work
- avoid retries that amplify backlog
- classify whether one subtree or many are affected

## Cache or Secondary Store Failing

- decide fail-open vs fail-closed explicitly
- protect origin from fallback amplification
- preserve critical workflows first

## Queue or Broker Degraded

- pause toxic consumers if replay worsens lag
- keep request path isolated from backlog pressure
- preserve correctness before throughput optics

## Agent Heuristics

- identify which process family is absorbing failure
- ask whether dependency pain is local, subtree-wide, or regional
- ask what signal proves containment is reducing mailbox pressure
