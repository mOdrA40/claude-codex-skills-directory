# RabbitMQ Troubleshooting Guide

## Quick Diagnosis Commands

```bash
# Overall health
rabbitmq-diagnostics check_running
rabbitmq-diagnostics check_local_alarms
rabbitmq-diagnostics check_port_connectivity
rabbitmq-diagnostics node_health_check

# Cluster status
rabbitmqctl cluster_status
rabbitmqctl list_alarms

# Resource usage
rabbitmqctl status | grep -A20 "Memory"
rabbitmqctl status | grep -A10 "File descriptors"
rabbitmqctl eval 'erlang:memory().'

# Queue issues
rabbitmqctl list_queues name messages consumers memory state
rabbitmqctl list_queues name messages_ready messages_unacknowledged

# Connection issues
rabbitmqctl list_connections name state timeout user vhost
rabbitmqctl list_connections pid name peer_host peer_port state
```

## Problem: Memory Alarm

### Symptoms
- Publishers blocked
- `rabbitmqctl list_alarms` shows `{memory, {resource_limit, memory, ...}}`
- Management UI shows red memory indicator

### Diagnosis

```bash
# Check memory breakdown
rabbitmqctl status | grep -A30 "Memory"

# Or detailed:
rabbitmqctl eval 'rabbit_vm:memory().'

# Key memory consumers:
# - binary: Message content in memory
# - queue_procs: Queue process overhead
# - connection_procs: Connection overhead
# - msg_index: Queue index
# - plugins: Plugin memory
```

### Solutions

```bash
# 1. Immediate: Increase memory limit (temporary)
rabbitmqctl set_vm_memory_high_watermark 0.7

# 2. Find memory hogs
rabbitmqctl list_queues name messages memory --sort memory --reversed --limit 10

# 3. Purge unnecessary queues
rabbitmqctl purge_queue <queue_name>

# 4. Enable lazy queues (for large queues)
rabbitmqctl set_policy lazy-queues "^large-" '{"queue-mode":"lazy"}' --apply-to queues

# 5. Long-term: Tune consumers
# - Increase prefetch to reduce queue overhead
# - Add more consumers
# - Use quorum queues (more efficient memory)

# 6. Configuration change
# rabbitmq.conf
vm_memory_high_watermark.relative = 0.6
vm_memory_high_watermark_paging_ratio = 0.5
```

### Root Cause Investigation

```python
# Script to analyze queue memory usage
import subprocess
import json

result = subprocess.run(
    ['rabbitmqctl', 'list_queues', 'name', 'messages', 'memory', 
     'consumers', 'message_bytes', '--formatter=json'],
    capture_output=True
)

queues = json.loads(result.stdout)

# Sort by memory
queues.sort(key=lambda q: q['memory'], reverse=True)

print("Top memory consumers:")
for q in queues[:10]:
    memory_mb = q['memory'] / 1024 / 1024
    msg_bytes_mb = q['message_bytes'] / 1024 / 1024
    overhead = memory_mb - msg_bytes_mb
    
    print(f"{q['name']}: {memory_mb:.2f}MB total, {msg_bytes_mb:.2f}MB messages, "
          f"{overhead:.2f}MB overhead, {q['messages']} msgs, {q['consumers']} consumers")
```

## Problem: Disk Alarm

### Diagnosis

```bash
# Check disk status
rabbitmqctl status | grep "Disk free limit"
df -h /var/lib/rabbitmq

# Check what's using disk
du -sh /var/lib/rabbitmq/mnesia/*
```

### Solutions

```bash
# 1. Clear log files
truncate -s 0 /var/log/rabbitmq/*.log

# 2. Purge old message store segments
# WARNING: May cause message loss
rabbitmqctl eval 'rabbit_msg_store_gc:gc().'

# 3. Delete unused queues
rabbitmqctl list_queues name messages --formatter=json | \
  jq -r '.[] | select(.messages == 0) | .name' | \
  xargs -I {} rabbitmqctl delete_queue {}

# 4. Increase disk limit (if false alarm)
rabbitmqctl set_disk_free_limit 2GB

# 5. Add storage
```

## Problem: Network Partition

### Symptoms
- `rabbitmqctl cluster_status` shows partitions
- Some nodes unreachable
- Split-brain behavior

### Diagnosis

```bash
# Check partition status
rabbitmqctl cluster_status

# Example output with partition:
# Network Partitions
#   * Node rabbit@node2 cannot communicate with rabbit@node1

# Check node connectivity
rabbitmq-diagnostics check_port_connectivity

# Check Erlang distribution
rabbitmq-diagnostics erlang_cookie_hash
rabbitmq-diagnostics check_certificate_expiration
```

### Resolution

```bash
# Option 1: Restart minority partition (preferred)
# On the minority partition node(s):
rabbitmqctl stop_app
rabbitmqctl start_app

# Option 2: Force restart
rabbitmqctl stop_app
rabbitmqctl force_reset
rabbitmqctl join_cluster rabbit@surviving-node
rabbitmqctl start_app

# Option 3: Manual intervention
# If autoheal or pause_minority doesn't work:
# 1. Stop all nodes
# 2. Start the node with the most recent data first
# 3. Join other nodes to it

# Prevention
# In rabbitmq.conf:
cluster_partition_handling = pause_minority
net_ticktime = 60
```

## Problem: Consumer Starvation

### Symptoms
- Queue depth growing
- Consumer count > 0 but messages not delivered
- Consumer utilization < 1.0

### Diagnosis

```bash
# Check consumer utilization
rabbitmqctl list_queues name consumers consumer_utilisation messages_unacknowledged

# Low utilization + high unacked = slow consumers
# Low utilization + low unacked = prefetch too high

# Check specific consumer
rabbitmqctl list_consumers queue_name channel_pid ack_required prefetch_count
```

### Solutions

```python
# Problem: Prefetch too low - consumer waiting for messages
# Solution: Increase prefetch
channel.basic_qos(prefetch_count=100)  # Was 1

# Problem: Prefetch too high - memory issues
# Solution: Decrease prefetch
channel.basic_qos(prefetch_count=10)  # Was 1000

# Problem: Slow consumers
# Solution 1: Optimize processing
# Solution 2: Add more consumers
# Solution 3: Batch processing

# Problem: Consumer blocked on I/O
# Solution: Use async I/O or thread pool
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

def callback(ch, method, props, body):
    # Submit to thread pool
    future = executor.submit(process_message, body)
    future.add_done_callback(lambda f: ch.basic_ack(method.delivery_tag))
```

## Problem: Connection Churn

### Symptoms
- High connection open/close rate
- `rabbitmq_connections_opened_total` growing fast
- Performance degradation

### Diagnosis

```bash
# Check connection rate
rabbitmqctl list_connections name peer_host state timeout | head -20

# Check for short-lived connections
watch -n 1 'rabbitmqctl list_connections name state | wc -l'
```

### Solutions

```python
# Problem: Creating connection per message
# BAD
def send_message(msg):
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.basic_publish(...)
    conn.close()  # Creates new connection every time!

# GOOD: Use connection pool
from queue import Queue
import threading

class ConnectionPool:
    def __init__(self, params, size=10):
        self.params = params
        self.pool = Queue(maxsize=size)
        for _ in range(size):
            self.pool.put(self._create_connection())
            
    def _create_connection(self):
        return pika.BlockingConnection(self.params)
        
    def get_connection(self):
        return self.pool.get()
        
    def return_connection(self, conn):
        if conn.is_open:
            self.pool.put(conn)
        else:
            self.pool.put(self._create_connection())

# Use contextmanager
from contextlib import contextmanager

@contextmanager
def connection(pool):
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        pool.return_connection(conn)
```

## Problem: Unroutable Messages

### Symptoms
- `basic.return` received
- `rabbitmq_global_messages_unroutable_returned_total` increasing
- Messages disappearing

### Diagnosis

```bash
# Check exchange bindings
rabbitmqctl list_bindings source_name destination_name routing_key

# Check if exchange exists
rabbitmqctl list_exchanges name type
```

### Solutions

```python
# 1. Handle returns
def on_return(ch, method, props, body):
    print(f"Message returned: {method.routing_key}")
    # Re-route or store in fallback queue
    
channel.add_on_return_callback(on_return)

# 2. Use alternate exchange
channel.exchange_declare(
    'my-exchange',
    'direct',
    arguments={'alternate-exchange': 'unrouted-exchange'}
)

# Alternate exchange catches all unroutable messages
channel.exchange_declare('unrouted-exchange', 'fanout')
channel.queue_declare('unrouted-messages')
channel.queue_bind('unrouted-messages', 'unrouted-exchange')

# 3. Ensure bindings exist before publishing
```

## Problem: Message Loss

### Investigation Checklist

```markdown
1. [ ] Publisher confirms enabled?
2. [ ] Messages persistent (delivery_mode=2)?
3. [ ] Queue durable?
4. [ ] Consumer using manual ack?
5. [ ] Ack sent AFTER processing complete?
6. [ ] No unhandled exceptions before ack?
7. [ ] No auto_ack=True?
8. [ ] Transaction or confirm delivery mode?
9. [ ] Quorum queue for critical data?
10. [ ] HA policy applied (if classic mirrored)?
```

### Debug Message Flow

```python
# Enable publisher confirms with callback
def on_confirm(frame):
    if isinstance(frame, pika.frame.Method):
        if isinstance(frame.method, pika.spec.Basic.Ack):
            print(f"Message {frame.method.delivery_tag} confirmed")
        elif isinstance(frame.method, pika.spec.Basic.Nack):
            print(f"Message {frame.method.delivery_tag} NACKED!")

channel.confirm_delivery()
channel.add_on_ack_callback(on_confirm)

# Trace message through system
# 1. Enable firehose tracer
rabbitmqctl trace_on

# 2. Consume from amq.rabbitmq.trace
# Messages published: publish.<exchange>
# Messages delivered: deliver.<queue>
```

## Problem: Slow Performance

### Benchmark Current State

```bash
# Use PerfTest tool
rabbitmq-perf-test --uri amqp://localhost \
    --producers 1 --consumers 1 \
    --rate 10000 --time 60 \
    --size 1000

# Expected baseline:
# - Same machine: 20,000+ msg/s
# - LAN: 10,000+ msg/s
# - WAN: Variable (network limited)
```

### Common Bottlenecks

```markdown
1. **Persistent messages + fsync**
   - Solution: Use lazy queues, quorum queues, or batch confirms

2. **Low prefetch**
   - Solution: Increase prefetch_count

3. **Connection per message**
   - Solution: Connection pooling

4. **Small message batching**
   - Solution: Batch multiple messages before confirm

5. **Synchronous confirms**
   - Solution: Use async confirms

6. **Queue mirroring overhead**
   - Solution: Use quorum queues (better performance)

7. **Plugin overhead**
   - Solution: Disable unused plugins

8. **TLS overhead**
   - Solution: Use hardware acceleration, session resumption
```

### Optimization Script

```python
# Performance diagnosis script
import pika
import time

def benchmark(params, num_messages=10000, message_size=1000):
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare('benchmark', durable=True)
    ch.confirm_delivery()
    
    message = b'x' * message_size
    
    # Publish benchmark
    start = time.time()
    for i in range(num_messages):
        ch.basic_publish(
            exchange='',
            routing_key='benchmark',
            body=message,
            properties=pika.BasicProperties(delivery_mode=2)
        )
    publish_time = time.time() - start
    
    # Consume benchmark
    received = [0]
    start = time.time()
    
    def callback(ch, method, props, body):
        received[0] += 1
        ch.basic_ack(method.delivery_tag)
        if received[0] >= num_messages:
            ch.stop_consuming()
    
    ch.basic_qos(prefetch_count=100)
    ch.basic_consume('benchmark', callback)
    ch.start_consuming()
    consume_time = time.time() - start
    
    print(f"Publish: {num_messages/publish_time:.0f} msg/s")
    print(f"Consume: {num_messages/consume_time:.0f} msg/s")
    
    ch.queue_delete('benchmark')
    conn.close()
```

## Emergency Procedures

### Node Won't Start

```bash
# Check what's wrong
rabbitmq-diagnostics status

# Check logs
tail -100 /var/log/rabbitmq/rabbit*.log

# Common issues:
# 1. Erlang cookie mismatch
cat /var/lib/rabbitmq/.erlang.cookie
# Must match across cluster

# 2. Mnesia corruption
# WARNING: DATA LOSS
rm -rf /var/lib/rabbitmq/mnesia/rabbit@hostname/*
# Then rejoin cluster

# 3. Port in use
lsof -i :5672
lsof -i :25672

# 4. Permission issues
chown -R rabbitmq:rabbitmq /var/lib/rabbitmq
chmod 400 /var/lib/rabbitmq/.erlang.cookie
```

### Emergency Queue Drain

```bash
#!/bin/bash
# Emergency drain script - use when queue is causing issues

QUEUE=$1
BATCH_SIZE=1000

while true; do
    COUNT=$(rabbitmqctl list_queues name messages | grep "^$QUEUE" | awk '{print $2}')
    
    if [ "$COUNT" -eq 0 ]; then
        echo "Queue drained"
        break
    fi
    
    echo "Remaining: $COUNT messages"
    
    # Move messages to archive
    rabbitmqctl eval "
        {ok, Q} = rabbit_amqqueue:lookup(rabbit_misc:r(<<\"/\">>, queue, <<\"$QUEUE\">>)),
        rabbit_amqqueue:purge(Q)
    ."
    
    # Or use shovel to move to archive queue
done
```

### Force Remove Stuck Node

```bash
# If a node is permanently dead
rabbitmqctl forget_cluster_node rabbit@dead-node

# If stuck on offline node
rabbitmqctl forget_cluster_node rabbit@dead-node --offline
```

## Diagnostic Query Reference

```erlang
%% Run with: rabbitmqctl eval '<expression>.'

%% Memory breakdown
rabbit_vm:memory().

%% Process info
erlang:process_info(whereis(rabbit_sup)).

%% Queue process memory
[{N, erlang:process_info(P, memory)} || 
 {N, P} <- rabbit_amqqueue:list()].

%% Connection details
[{P, rabbit_reader:info(P, [peer_host, user, vhost, timeout])} || 
 {P, _} <- supervisor:which_children(rabbit_tcp_client_sup)].

%% Binding count per exchange
[{X, length(rabbit_binding:list_for_source(X))} || 
 X <- rabbit_exchange:list()].

%% Message store info
rabbit_msg_store:info(msg_store_persistent).
rabbit_msg_store:info(msg_store_transient).

%% Queue index info
[{Q, rabbit_queue_index:info(Q)} || Q <- rabbit_amqqueue:list_names()].
```
