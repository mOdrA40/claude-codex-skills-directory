# Outage Triage: First 15 Minutes on the BEAM

## First Questions

- did a deploy, node topology change, or dependency failure just happen?
- are mailboxes growing in one process family or broadly?
- is the bottleneck dependency latency, message backlog, scheduler pressure, or queue replay?
- is one supervision subtree creating the blast radius?

## Containment Options

- stop rollout or cluster topology change
- pause toxic consumers or fan-out paths
- shed optional work and protect critical request paths
- isolate or drain bad nodes if they amplify pressure
- reduce mailbox growth before hunting elegant fixes

## Avoid

- adding broad rescue blocks
- restarting everything without understanding which subtree is sick
- assuming concurrency equals resilience automatically
- coupling containment with risky architecture rewrites

## Agent Heuristic

For BEAM incidents, identify which process family is absorbing the pressure and whether the failure is local, subtree-wide, or regional.
