# RabbitMQ Clustering & High Availability

## Cluster Architecture

### Minimum Production Cluster

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│                   (HAProxy/Nginx/ALB)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │ Node 1  │◄──► Node 2  ◄───► Node 3  │
   │ (disc)  │   │ (disc)  │   │ (disc)  │
   └─────────┘   └─────────┘   └─────────┘
        │             │             │
   Quorum Queue Replicas (Raft consensus)
```

### Node Types

| Type | Data Storage | Use Case |
|------|--------------|----------|
| Disc | Persistent | Production (always use) |
| RAM | Memory only | NEVER in production |

**Rule: ALL production nodes must be disc nodes.**

## Cluster Setup

### 1. Prerequisites

```bash
# /etc/hosts on ALL nodes
192.168.1.10 rabbitmq-1
192.168.1.11 rabbitmq-2
192.168.1.12 rabbitmq-3

# Erlang cookie MUST be identical on all nodes
# /var/lib/rabbitmq/.erlang.cookie
# chmod 400 .erlang.cookie
# chown rabbitmq:rabbitmq .erlang.cookie
```

### 2. Cluster Formation

```bash
# On node 2 and 3
rabbitmqctl stop_app
rabbitmqctl reset
rabbitmqctl join_cluster rabbit@rabbitmq-1
rabbitmqctl start_app

# Verify
rabbitmqctl cluster_status
```

### 3. Production Configuration

```ini
# /etc/rabbitmq/rabbitmq.conf

# Cluster identity
cluster_name = prod-rabbitmq

# Clustering
cluster_formation.peer_discovery_backend = rabbit_peer_discovery_classic_config
cluster_formation.classic_config.nodes.1 = rabbit@rabbitmq-1
cluster_formation.classic_config.nodes.2 = rabbit@rabbitmq-2
cluster_formation.classic_config.nodes.3 = rabbit@rabbitmq-3

# Partition handling - CRITICAL
cluster_partition_handling = pause_minority

# Network tuning
net_ticktime = 60
distribution.buffer_size = 128000

# Quorum queues default
default_queue_type = quorum
```

## Network Partition Handling

### Partition Strategies

| Strategy | Behavior | When to Use |
|----------|----------|-------------|
| `pause_minority` | Minority partition stops | **RECOMMENDED** - prevents split brain |
| `pause_if_all_down` | Pause if can't reach specified nodes | Specific topology requirements |
| `autoheal` | Auto-recover, may lose data | Non-critical data, prioritize availability |
| `ignore` | Do nothing | NEVER in production |

### Detecting Partitions

```bash
# Check partition status
rabbitmqctl cluster_status

# Look for:
# Network Partitions
# Node rabbit@node2 cannot communicate with rabbit@node1

# Alarms
rabbitmqctl list_alarms
```

### Recovering from Partitions

```bash
# Option 1: Restart minority nodes
# On minority partition nodes:
rabbitmqctl stop_app
rabbitmqctl start_app

# Option 2: Force reset (DATA LOSS on this node)
rabbitmqctl stop_app
rabbitmqctl force_reset
rabbitmqctl join_cluster rabbit@surviving-node
rabbitmqctl start_app

# Option 3: Manual partition resolution
rabbitmqctl forget_cluster_node rabbit@dead-node --offline
```

## Quorum Queues (Recommended)

### Why Quorum Queues?

| Feature | Classic Mirrored | Quorum |
|---------|------------------|--------|
| Replication | Async | Raft consensus |
| Data safety | May lose on failure | Strong consistency |
| Partition behavior | Split-brain risk | Leader election |
| Performance | Higher throughput | Higher durability |
| Memory | Lower | Higher (Raft log) |

**Always use Quorum queues for critical data.**

### Quorum Queue Configuration

```python
channel.queue_declare(
    'orders',
    durable=True,
    arguments={
        'x-queue-type': 'quorum',
        
        # Replication
        'x-quorum-initial-group-size': 3,  # Match cluster size
        
        # Delivery limits (auto-DLQ after N redeliveries)
        'x-delivery-limit': 5,
        
        # Memory management
        'x-max-in-memory-length': 10000,  # Spill to disk after
        'x-max-in-memory-bytes': 104857600,  # 100MB
        
        # Dead lettering
        'x-dead-letter-exchange': 'dlx',
        'x-dead-letter-routing-key': 'orders.dead',
        'x-dead-letter-strategy': 'at-least-once'
    }
)
```

### Quorum Queue Limitations

- No priority queues
- No lazy queue mode (built-in instead)
- No message TTL per-message (queue-level only)
- No global QoS
- Higher memory footprint

## Stream Queues (High Throughput)

### When to Use Streams

- Event sourcing / audit logs
- High throughput (millions/sec)
- Message replay needed
- Multiple consumers reading same data
- Time-travel queries

### Stream Configuration

```python
channel.queue_declare(
    'events.stream',
    durable=True,
    arguments={
        'x-queue-type': 'stream',
        
        # Retention
        'x-max-length-bytes': 50_000_000_000,  # 50GB
        'x-max-age': '7D',  # 7 days
        
        # Segment size (affects replay performance)
        'x-stream-max-segment-size-bytes': 500_000_000,  # 500MB
        
        # Filter (server-side filtering)
        'x-stream-filter-size-bytes': 16  # Bloom filter size
    }
)

# Consume with offset
channel.basic_consume(
    'events.stream',
    callback,
    arguments={
        # 'first' - from beginning
        # 'last' - from end
        # 'next' - only new messages
        # timestamp - from specific time
        # offset - from specific offset
        'x-stream-offset': 'first',
        
        # Or timestamp-based
        # 'x-stream-offset': datetime(2024, 1, 1).timestamp()
        
        # Or specific offset
        # 'x-stream-offset': 12345
    }
)
```

## Federation & Shovel

### When to Use

| Use Case | Solution |
|----------|----------|
| WAN replication | Federation |
| Datacenter failover | Federation |
| Queue migration | Shovel |
| One-time data transfer | Shovel |
| Continuous replication | Federation |

### Federation Setup

```bash
# Enable plugin
rabbitmq-plugins enable rabbitmq_federation
rabbitmq-plugins enable rabbitmq_federation_management

# Define upstream
rabbitmqctl set_parameter federation-upstream dc1 \
  '{"uri":"amqp://user:pass@dc1-rabbitmq:5672","expires":3600000}'

# Create policy
rabbitmqctl set_policy federate-orders \
  "^orders\." \
  '{"federation-upstream-set":"all"}' \
  --priority 10 \
  --apply-to exchanges
```

### Federation Configuration

```ini
# rabbitmq.conf

# Federation link tuning
federation.default.ack_mode = on-confirm
federation.default.prefetch_count = 1000
federation.default.reconnect_delay = 5
federation.default.trust_user_id = false
```

### Shovel Setup

```bash
# Enable plugin
rabbitmq-plugins enable rabbitmq_shovel
rabbitmq-plugins enable rabbitmq_shovel_management

# Create dynamic shovel
rabbitmqctl set_parameter shovel migrate-orders \
'{
  "src-protocol": "amqp091",
  "src-uri": "amqp://old-cluster",
  "src-queue": "orders",
  "dest-protocol": "amqp091", 
  "dest-uri": "amqp://new-cluster",
  "dest-queue": "orders",
  "ack-mode": "on-confirm",
  "prefetch-count": 1000,
  "reconnect-delay": 5,
  "delete-after": "queue-length"
}'
```

## Load Balancing

### HAProxy Configuration

```haproxy
# /etc/haproxy/haproxy.cfg

global
    maxconn 50000
    
defaults
    mode tcp
    timeout connect 5s
    timeout client 120s
    timeout server 120s
    
# AMQP load balancing
frontend rabbitmq_amqp
    bind *:5672
    default_backend rabbitmq_amqp_backend
    
backend rabbitmq_amqp_backend
    balance roundrobin
    option tcp-check
    
    # Health check on AMQP port
    server rabbitmq-1 192.168.1.10:5672 check inter 5s rise 2 fall 3
    server rabbitmq-2 192.168.1.11:5672 check inter 5s rise 2 fall 3
    server rabbitmq-3 192.168.1.12:5672 check inter 5s rise 2 fall 3

# Management UI
frontend rabbitmq_mgmt
    bind *:15672
    default_backend rabbitmq_mgmt_backend
    
backend rabbitmq_mgmt_backend
    balance roundrobin
    option httpchk GET /api/health/checks/alarms
    http-check expect status 200
    
    server rabbitmq-1 192.168.1.10:15672 check inter 10s
    server rabbitmq-2 192.168.1.11:15672 check inter 10s
    server rabbitmq-3 192.168.1.12:15672 check inter 10s

# Stats
listen stats
    bind *:8404
    mode http
    stats enable
    stats uri /stats
    stats refresh 10s
```

### Nginx Stream Configuration

```nginx
# /etc/nginx/nginx.conf

stream {
    upstream rabbitmq {
        least_conn;
        server 192.168.1.10:5672 weight=5;
        server 192.168.1.11:5672 weight=5;
        server 192.168.1.12:5672 weight=5;
    }
    
    server {
        listen 5672;
        proxy_pass rabbitmq;
        proxy_connect_timeout 5s;
        proxy_timeout 120s;
    }
}
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup_rabbitmq.sh

BACKUP_DIR="/backup/rabbitmq/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Export definitions (exchanges, queues, bindings, users, policies)
rabbitmqctl export_definitions "$BACKUP_DIR/definitions.json"

# Backup messages (requires rabbitmq_management plugin)
# Note: This is NOT atomic - for consistent backups, consider:
# 1. Quorum queue snapshots
# 2. Shovel to backup cluster
# 3. Stream queue with long retention

# Backup mnesia data (requires node stop)
# cp -r /var/lib/rabbitmq/mnesia "$BACKUP_DIR/mnesia"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Upload to S3/GCS
aws s3 cp "$BACKUP_DIR.tar.gz" "s3://backups/rabbitmq/"
```

### Restore Procedure

```bash
#!/bin/bash
# restore_rabbitmq.sh

BACKUP_FILE=$1

# Stop all nodes
rabbitmqctl stop_app

# Reset cluster
rabbitmqctl force_reset

# Start fresh
rabbitmqctl start_app

# Import definitions
rabbitmqctl import_definitions "$BACKUP_FILE"

# Note: Messages are NOT restored - use shovel/federation for message DR
```

### Multi-DC Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         Global DNS                              │
│                    (Latency-based routing)                      │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │  DC 1   │◄───────►│  DC 2   │◄───────►│  DC 3   │
   │ Primary │Federation│ Primary │Federation│ Standby │
   │ (US-E)  │         │ (EU)    │         │ (APAC)  │
   └─────────┘         └─────────┘         └─────────┘
   
   Each DC: 3-node quorum cluster
   Federation: Bidirectional for active-active
   Standby: Passive, federation consumer only
```

### Failover Runbook

```markdown
# RabbitMQ Datacenter Failover Runbook

## Pre-requisites
- [ ] DNS TTL lowered to 60s (done during incident prep)
- [ ] Federation links verified healthy
- [ ] Backup cluster tested within last 24h

## Failover Steps

1. **Confirm Primary DC is down**
   - Check monitoring dashboards
   - Verify network connectivity
   - Confirm with NOC

2. **Promote Secondary DC**
   ```bash
   # On secondary DC leader
   rabbitmqctl set_policy failover-primary ".*" \
     '{"ha-mode":"all"}' --priority 100
   ```

3. **Update DNS**
   - Route53/CloudFlare: Update A records to secondary DC IPs
   - Verify propagation: `dig rabbitmq.prod.company.com`

4. **Notify Applications**
   - Applications should reconnect automatically
   - If not, trigger rolling restart

5. **Monitor**
   - Watch connection counts
   - Monitor queue depths
   - Check consumer lag

## Failback Steps
(After primary DC recovered)

1. Restore federation from secondary to primary
2. Verify data sync complete
3. Gradually shift traffic back
4. Restore original DNS weights
```
