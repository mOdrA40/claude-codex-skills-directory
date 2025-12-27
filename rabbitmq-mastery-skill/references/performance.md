# RabbitMQ Performance Tuning

## Performance Fundamentals

### Message Flow Pipeline

```
Publisher → Network → Channel → Exchange → Binding → Queue → Channel → Network → Consumer
           ↓          ↓          ↓          ↓         ↓        ↓          ↓
        Latency    Credits    Routing    Lookup    Storage  Prefetch  Latency
```

### Performance Factors

| Factor | Impact | Tunable |
|--------|--------|---------|
| Message persistence | 10-100x slower | Yes |
| Publisher confirms | 2-5x slower (sync) | Yes |
| Message size | Linear throughput decrease | Design |
| Queue depth | Memory/disk pressure | Indirect |
| Consumer count | Better parallelism | Yes |
| Prefetch | Massive impact | Yes |
| Network latency | Dominates at scale | Infrastructure |

## Benchmarking

### Expected Baselines

```markdown
# Single node, localhost, 1KB messages

| Scenario | Throughput | p99 Latency |
|----------|------------|-------------|
| Transient, no confirms | 80,000+ msg/s | < 1ms |
| Transient, batch confirms (100) | 50,000+ msg/s | < 5ms |
| Persistent, batch confirms | 20,000+ msg/s | < 10ms |
| Persistent, sync confirms | 2,000 msg/s | < 50ms |
| Quorum queue, confirms | 15,000+ msg/s | < 15ms |
```

## Server Configuration

### Production rabbitmq.conf

```ini
# Memory
vm_memory_high_watermark.relative = 0.6
vm_memory_high_watermark_paging_ratio = 0.5

# Disk
disk_free_limit.absolute = 5GB

# Network
tcp_listen_options.backlog = 4096
tcp_listen_options.nodelay = true
tcp_listen_options.sndbuf = 196608
tcp_listen_options.recbuf = 196608

# Erlang VM
+P 1000000
+Q 1000000

# Clustering
cluster_partition_handling = pause_minority
net_ticktime = 60

# Default queue type
default_queue_type = quorum

# Flow control (advanced)
credit_flow_default_credit = {400, 200}
```

### OS Tuning

```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# File descriptors
fs.file-max = 1000000

# /etc/security/limits.conf
rabbitmq soft nofile 65536
rabbitmq hard nofile 65536
```

## Prefetch Optimization

```python
"""
Prefetch = unacked messages consumer can have buffered

Too low: Consumer starving, waiting for messages
Too high: Memory issues, uneven load, long recovery

Formula:
  optimal = (processing_time_ms / network_rtt_ms) * 1.5

Examples:
  - Local (1ms RTT), 50ms processing: 75
  - Same DC (5ms RTT), 50ms processing: 15  
  - Cross DC (50ms RTT), 50ms processing: 2
"""

# Auto-tuning implementation
class AdaptiveConsumer:
    def __init__(self, channel):
        self.channel = channel
        self.times = []
        self.prefetch = 50
        
    def process(self, ch, method, props, body):
        start = time.time()
        try:
            self._work(body)
            ch.basic_ack(method.delivery_tag)
        finally:
            self.times.append(time.time() - start)
            if len(self.times) % 100 == 0:
                self._tune()
                
    def _tune(self):
        avg = sum(self.times[-100:]) / 100
        optimal = max(10, min(500, int(avg / 0.005 * 1.5)))
        if abs(optimal - self.prefetch) > self.prefetch * 0.2:
            self.prefetch = optimal
            self.channel.basic_qos(prefetch_count=optimal)
```

## Batch Publishing

```python
def batch_publish(channel, messages, batch_size=100):
    """10x faster than individual confirms"""
    channel.confirm_delivery()
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        for msg in batch:
            channel.basic_publish(
                exchange=msg['exchange'],
                routing_key=msg['key'],
                body=msg['body'],
                properties=pika.BasicProperties(delivery_mode=2)
            )
        channel.wait_for_confirms(timeout=30)
```

## Queue Type Performance

### Quorum Queue Tuning

```python
channel.queue_declare('orders', durable=True, arguments={
    'x-queue-type': 'quorum',
    'x-quorum-initial-group-size': 3,
    'x-max-in-memory-length': 10000,
    'x-max-in-memory-bytes': 104857600,
    'x-delivery-limit': 5,
})
```

### Stream Queue Tuning  

```python
channel.queue_declare('events', durable=True, arguments={
    'x-queue-type': 'stream',
    'x-max-length-bytes': 50_000_000_000,
    'x-max-age': '7D',
    'x-stream-max-segment-size-bytes': 500_000_000,
})
```

## Message Size Optimization

```markdown
| Size | Throughput Impact | Recommendation |
|------|-------------------|----------------|
| < 4KB | Optimal | Direct in message |
| 4-64KB | Good | Consider compression |
| 64KB-1MB | Degraded | External storage + reference |
| > 1MB | Poor | Always external storage |
```

```python
import zlib

def publish_optimized(channel, body, threshold=4096):
    if len(body) > threshold:
        compressed = zlib.compress(body)
        if len(compressed) < len(body) * 0.8:
            body = compressed
            headers = {'compression': 'zlib'}
        else:
            headers = {}
    else:
        headers = {}
        
    channel.basic_publish(
        exchange='',
        routing_key='queue',
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,
            headers=headers
        )
    )
```

## Connection Pooling

```python
from queue import Queue
import threading

class ConnectionPool:
    def __init__(self, params, size=10):
        self.params = params
        self.pool = Queue(maxsize=size)
        self.lock = threading.Lock()
        
        for _ in range(size):
            self.pool.put(self._create())
            
    def _create(self):
        conn = pika.BlockingConnection(self.params)
        return {'connection': conn, 'channel': conn.channel()}
        
    def acquire(self):
        return self.pool.get()
        
    def release(self, item):
        if item['connection'].is_open:
            self.pool.put(item)
        else:
            self.pool.put(self._create())
```

## Performance Checklist

```markdown
## Publisher
- [ ] Connection pooling (not per-message)
- [ ] Batch confirms (100+ messages)
- [ ] Persistent only when needed
- [ ] Message compression for large payloads
- [ ] Multiple channels for parallelism

## Consumer  
- [ ] Prefetch tuned (not 1, not unlimited)
- [ ] Manual ack after processing
- [ ] Parallel processing (thread pool)
- [ ] Graceful shutdown (drain in-flight)

## Queue
- [ ] Quorum for durability
- [ ] Stream for high throughput/replay
- [ ] TTL to prevent bloat
- [ ] Max-length with overflow policy

## Server
- [ ] Memory watermark tuned
- [ ] Disk alarm threshold set
- [ ] File descriptors increased
- [ ] TCP buffers optimized
```
