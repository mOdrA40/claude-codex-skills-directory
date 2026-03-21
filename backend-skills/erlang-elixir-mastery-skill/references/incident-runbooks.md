# Incident Runbooks for Erlang / Elixir Systems

## Purpose

A principal-quality backend skill should not stop at design advice. It should teach what to do when the system is already failing.

## Runbook: Mailbox Growth

Symptoms:

- request latency rises
- one or more GenServers become bottlenecks
- memory grows or throughput collapses

Immediate actions:

- identify process or subsystem with abnormal mailbox length
- reduce or pause upstream fan-in if possible
- shed optional traffic
- scale consumer capacity only if the bottleneck is parallelizable
- inspect whether one process is doing too much coordination

## Runbook: Dependency Slowness

Symptoms:

- timeouts spike
- pool saturation appears
- retries increase system load

Immediate actions:

- confirm which dependency is slow
- disable non-critical downstream features
- reduce retry amplification
- tighten or lower concurrency to protect the rest of the service
- consider degraded mode instead of waiting for full failure

## Runbook: Restart Storm

Symptoms:

- supervisors restart children repeatedly
- throughput falls while CPU remains busy
- logs are noisy but uninformative

Immediate actions:

- identify the failing child and root dependency
- reduce traffic to the failing path
- decide whether restart intensity is amplifying the incident
- isolate the blast radius if the supervisor boundary is too broad

## Runbook: Netsplit or Cluster Instability

Symptoms:

- duplicate jobs
- inconsistent cluster views
- leadership confusion

Immediate actions:

- identify which nodes disagree
- disable globally unique operations if ownership is uncertain
- protect critical write paths
- prefer safe degradation over pretending the cluster is healthy

## Review Questions

- Can operators identify the failing subsystem in minutes?
- Can the system degrade safely?
- Which actions reduce blast radius fastest?
- Which features should be disabled first during stress?
