# RabbitMQ Monitoring & Observability

## Prometheus + Grafana Setup

### 1. Enable Prometheus Plugin

```bash
rabbitmq-plugins enable rabbitmq_prometheus
```

### 2. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rabbitmq'
    static_configs:
      - targets:
        - 'rabbitmq-1:15692'
        - 'rabbitmq-2:15692'
        - 'rabbitmq-3:15692'
    metrics_path: /metrics
    
  # Detailed queue metrics (expensive, sample less frequently)
  - job_name: 'rabbitmq-detailed'
    scrape_interval: 60s
    static_configs:
      - targets:
        - 'rabbitmq-1:15692'
    metrics_path: /metrics/detailed
    params:
      family:
        - queue_coarse_metrics
        - queue_metrics
```

### 3. Critical Metrics Reference

```yaml
# Node Health
rabbitmq_process_resident_memory_bytes      # Memory usage
rabbitmq_resident_memory_limit_bytes         # Memory limit
rabbitmq_disk_space_available_bytes          # Free disk
rabbitmq_disk_space_available_limit_bytes    # Disk limit
rabbitmq_fd_used                             # File descriptors used
rabbitmq_fd_available                        # File descriptors available
rabbitmq_sockets_used                        # TCP sockets
rabbitmq_erlang_processes_used               # Erlang processes

# Connections & Channels
rabbitmq_connections                         # Total connections
rabbitmq_connections_opened_total            # Connection rate
rabbitmq_connections_closed_total            # Close rate
rabbitmq_channels                            # Total channels
rabbitmq_consumers                           # Total consumers

# Message Flow
rabbitmq_global_messages_received_total      # Incoming messages
rabbitmq_global_messages_delivered_total     # Delivered messages
rabbitmq_global_messages_confirmed_total     # Publisher confirms
rabbitmq_global_messages_routed_total        # Routed messages
rabbitmq_global_messages_unroutable_returned_total  # Unroutable

# Queue Metrics (per queue)
rabbitmq_queue_messages_ready                # Ready for delivery
rabbitmq_queue_messages_unacknowledged       # In-flight
rabbitmq_queue_messages                      # Total (ready + unacked)
rabbitmq_queue_messages_published_total      # Publish rate
rabbitmq_queue_messages_delivered_total      # Delivery rate
rabbitmq_queue_messages_acked_total          # Ack rate
rabbitmq_queue_messages_redelivered_total    # Redelivery rate
rabbitmq_queue_consumer_utilisation          # Consumer efficiency
rabbitmq_queue_consumers                     # Consumer count
rabbitmq_queue_messages_bytes                # Queue size in bytes

# Cluster
rabbitmq_cluster_size                        # Nodes in cluster
rabbitmq_cluster_partitions                  # Network partitions
```

### 4. Grafana Dashboard (JSON)

```json
{
  "title": "RabbitMQ Production Dashboard",
  "panels": [
    {
      "title": "Memory Usage %",
      "type": "gauge",
      "targets": [{
        "expr": "rabbitmq_process_resident_memory_bytes / rabbitmq_resident_memory_limit_bytes * 100",
        "legendFormat": "{{instance}}"
      }],
      "thresholds": [60, 80]
    },
    {
      "title": "Messages Ready (All Queues)",
      "type": "graph",
      "targets": [{
        "expr": "sum(rabbitmq_queue_messages_ready) by (queue)",
        "legendFormat": "{{queue}}"
      }]
    },
    {
      "title": "Message Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(rabbitmq_global_messages_received_total[5m])",
          "legendFormat": "Received/s"
        },
        {
          "expr": "rate(rabbitmq_global_messages_delivered_total[5m])",
          "legendFormat": "Delivered/s"
        }
      ]
    },
    {
      "title": "Consumer Utilization",
      "type": "graph",
      "targets": [{
        "expr": "rabbitmq_queue_consumer_utilisation",
        "legendFormat": "{{queue}}"
      }]
    },
    {
      "title": "Connections",
      "type": "stat",
      "targets": [{
        "expr": "sum(rabbitmq_connections)"
      }]
    },
    {
      "title": "Unacknowledged Messages",
      "type": "graph",
      "targets": [{
        "expr": "sum(rabbitmq_queue_messages_unacknowledged) by (queue)",
        "legendFormat": "{{queue}}"
      }]
    }
  ]
}
```

## AlertManager Rules

```yaml
# rabbitmq_alerts.yml
groups:
  - name: rabbitmq.critical
    rules:
      # Memory alarm
      - alert: RabbitMQMemoryHigh
        expr: rabbitmq_process_resident_memory_bytes / rabbitmq_resident_memory_limit_bytes > 0.8
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "RabbitMQ memory usage above 80%"
          description: "{{ $labels.instance }}: {{ $value | humanizePercentage }}"
          runbook: "https://wiki/rabbitmq/memory-alarm"
          
      # Disk alarm
      - alert: RabbitMQDiskLow
        expr: rabbitmq_disk_space_available_bytes / 1e9 < 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "RabbitMQ disk space below 5GB"
          
      # Network partition
      - alert: RabbitMQNetworkPartition
        expr: rabbitmq_cluster_partitions > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "RabbitMQ network partition detected"
          runbook: "https://wiki/rabbitmq/partition-recovery"
          
      # Node down
      - alert: RabbitMQNodeDown
        expr: up{job="rabbitmq"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RabbitMQ node {{ $labels.instance }} is down"
          
      # Cluster size reduced
      - alert: RabbitMQClusterSmall
        expr: rabbitmq_cluster_size < 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RabbitMQ cluster has less than 3 nodes"

  - name: rabbitmq.queue
    rules:
      # Queue backlog
      - alert: RabbitMQQueueBacklog
        expr: rabbitmq_queue_messages_ready > 100000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Queue {{ $labels.queue }} has {{ $value }} messages"
          
      # Consumer starvation
      - alert: RabbitMQNoConsumers
        expr: rabbitmq_queue_consumers == 0 and rabbitmq_queue_messages > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Queue {{ $labels.queue }} has no consumers"
          
      # Low consumer utilization
      - alert: RabbitMQLowConsumerUtilization
        expr: rabbitmq_queue_consumer_utilisation < 0.5 and rabbitmq_queue_messages_ready > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Queue {{ $labels.queue }} consumers underutilized"
          description: "Utilization: {{ $value | humanizePercentage }}. Consider reducing prefetch or adding consumers."
          
      # Unacked messages growing
      - alert: RabbitMQHighUnackedMessages
        expr: rabbitmq_queue_messages_unacknowledged > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.queue }} has {{ $value }} unacked messages"
          description: "Consumers may be slow or stuck. Check consumer health."
          
      # Message redelivery rate
      - alert: RabbitMQHighRedeliveryRate
        expr: rate(rabbitmq_queue_messages_redelivered_total[5m]) / rate(rabbitmq_queue_messages_delivered_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High redelivery rate on {{ $labels.queue }}"
          description: "{{ $value | humanizePercentage }} of messages being redelivered"

  - name: rabbitmq.connections
    rules:
      # Connection churn
      - alert: RabbitMQHighConnectionChurn
        expr: rate(rabbitmq_connections_opened_total[5m]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High connection open rate: {{ $value }}/s"
          description: "Applications may be creating too many short-lived connections"
          
      # Near connection limit
      - alert: RabbitMQConnectionsHigh
        expr: rabbitmq_connections / rabbitmq_fd_available > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Connections approaching file descriptor limit"

  - name: rabbitmq.publishing
    rules:
      # Unroutable messages
      - alert: RabbitMQUnroutableMessages
        expr: rate(rabbitmq_global_messages_unroutable_returned_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $value }}/s unroutable messages"
          description: "Check exchange/queue bindings"
          
      # Publisher confirm failures
      - alert: RabbitMQPublishNacks
        expr: rate(rabbitmq_global_messages_confirmed_total[5m]) < rate(rabbitmq_global_messages_received_total[5m]) * 0.99
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Publisher confirms failing"
```

## Custom Health Check Endpoint

```python
# health_check_server.py
from flask import Flask, jsonify
import pika
import subprocess
import json

app = Flask(__name__)

@app.route('/health')
def health():
    checks = {
        'rabbitmq_running': check_rabbitmq_running(),
        'cluster_healthy': check_cluster_status(),
        'no_alarms': check_alarms(),
        'memory_ok': check_memory(),
        'disk_ok': check_disk(),
        'queues_healthy': check_queue_health(),
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks
    }), status_code

@app.route('/health/ready')
def readiness():
    """Kubernetes readiness probe"""
    try:
        conn = pika.BlockingConnection(
            pika.ConnectionParameters('localhost', socket_timeout=5)
        )
        conn.close()
        return jsonify({'status': 'ready'}), 200
    except:
        return jsonify({'status': 'not ready'}), 503

@app.route('/health/live')
def liveness():
    """Kubernetes liveness probe"""
    result = subprocess.run(
        ['rabbitmqctl', 'status'],
        capture_output=True,
        timeout=10
    )
    if result.returncode == 0:
        return jsonify({'status': 'alive'}), 200
    return jsonify({'status': 'dead'}), 503

def check_rabbitmq_running():
    result = subprocess.run(['rabbitmqctl', 'status'], capture_output=True)
    return result.returncode == 0

def check_cluster_status():
    result = subprocess.run(
        ['rabbitmqctl', 'cluster_status', '--formatter=json'],
        capture_output=True
    )
    if result.returncode != 0:
        return False
    data = json.loads(result.stdout)
    # Check no partitions
    return len(data.get('partitions', [])) == 0

def check_alarms():
    result = subprocess.run(
        ['rabbitmqctl', 'list_alarms', '--formatter=json'],
        capture_output=True
    )
    if result.returncode != 0:
        return False
    alarms = json.loads(result.stdout)
    return len(alarms) == 0

def check_memory():
    result = subprocess.run(
        ['rabbitmqctl', 'eval', 'vm_memory_monitor:get_memory_use().'],
        capture_output=True
    )
    # Returns {memory_use, memory_limit}
    return b'alarm' not in result.stdout.lower()

def check_disk():
    result = subprocess.run(
        ['rabbitmqctl', 'eval', 'rabbit_disk_monitor:get_disk_free_limit().'],
        capture_output=True
    )
    return result.returncode == 0

def check_queue_health():
    """Check for stuck or oversized queues"""
    result = subprocess.run(
        ['rabbitmqctl', 'list_queues', 'name', 'messages', 'consumers', '--formatter=json'],
        capture_output=True
    )
    if result.returncode != 0:
        return False
    
    queues = json.loads(result.stdout)
    for q in queues:
        # Queue with messages but no consumers
        if q['messages'] > 10000 and q['consumers'] == 0:
            return False
        # Queue too large
        if q['messages'] > 1000000:
            return False
    return True

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Distributed Tracing (OpenTelemetry)

```python
# traced_consumer.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import pika
import json

# Setup tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name='jaeger',
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
tracer = trace.get_tracer(__name__)
propagator = TraceContextTextMapPropagator()


class TracedPublisher:
    def __init__(self, channel):
        self.channel = channel
        
    def publish(self, exchange, routing_key, body, parent_span=None):
        with tracer.start_as_current_span(
            f"publish:{routing_key}",
            kind=trace.SpanKind.PRODUCER
        ) as span:
            span.set_attribute("messaging.system", "rabbitmq")
            span.set_attribute("messaging.destination", exchange)
            span.set_attribute("messaging.routing_key", routing_key)
            span.set_attribute("messaging.message.payload_size_bytes", len(body))
            
            # Inject trace context into headers
            headers = {}
            inject(headers, setter=lambda d, k, v: d.update({k: v}))
            
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    headers=headers
                )
            )


class TracedConsumer:
    def __init__(self, channel):
        self.channel = channel
        
    def process_message(self, ch, method, props, body):
        # Extract trace context from headers
        headers = props.headers or {}
        ctx = extract(headers, getter=lambda d, k: d.get(k))
        
        with tracer.start_as_current_span(
            f"process:{method.routing_key}",
            context=ctx,
            kind=trace.SpanKind.CONSUMER
        ) as span:
            span.set_attribute("messaging.system", "rabbitmq")
            span.set_attribute("messaging.destination", method.exchange)
            span.set_attribute("messaging.routing_key", method.routing_key)
            span.set_attribute("messaging.message.payload_size_bytes", len(body))
            
            try:
                self._do_work(body)
                span.set_status(trace.Status(trace.StatusCode.OK))
                ch.basic_ack(method.delivery_tag)
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                ch.basic_nack(method.delivery_tag, requeue=False)
```

## Log Aggregation

```python
# structured_logging.py
import logging
import json
from datetime import datetime

class RabbitMQJsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': 'order-consumer',
            'environment': 'production',
        }
        
        # Add extra fields
        if hasattr(record, 'queue'):
            log_record['queue'] = record.queue
        if hasattr(record, 'message_id'):
            log_record['message_id'] = record.message_id
        if hasattr(record, 'delivery_tag'):
            log_record['delivery_tag'] = record.delivery_tag
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        if hasattr(record, 'processing_time_ms'):
            log_record['processing_time_ms'] = record.processing_time_ms
            
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)


# Usage
logger = logging.getLogger('rabbitmq.consumer')
handler = logging.StreamHandler()
handler.setFormatter(RabbitMQJsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# In consumer
def process_message(ch, method, props, body):
    start = time.time()
    try:
        result = do_work(body)
        logger.info(
            "Message processed successfully",
            extra={
                'queue': 'orders',
                'message_id': props.message_id,
                'delivery_tag': method.delivery_tag,
                'correlation_id': props.correlation_id,
                'processing_time_ms': (time.time() - start) * 1000
            }
        )
    except Exception as e:
        logger.exception(
            "Message processing failed",
            extra={
                'queue': 'orders',
                'message_id': props.message_id,
                'error_type': type(e).__name__
            }
        )
```

## SLA Monitoring

```python
# sla_monitor.py
"""
Track message processing SLAs:
- End-to-end latency (publish to ack)
- Processing time
- Error rates
- Throughput
"""

from prometheus_client import Histogram, Counter, Gauge, start_http_server
import time

# Metrics
MESSAGE_LATENCY = Histogram(
    'message_latency_seconds',
    'End-to-end message latency',
    ['queue', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60]
)

PROCESSING_TIME = Histogram(
    'message_processing_seconds',
    'Message processing time',
    ['queue', 'handler'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
)

MESSAGES_PROCESSED = Counter(
    'messages_processed_total',
    'Total messages processed',
    ['queue', 'status']
)

MESSAGES_IN_FLIGHT = Gauge(
    'messages_in_flight',
    'Currently processing messages',
    ['queue']
)


class SLATrackedConsumer:
    def __init__(self, channel, queue_name):
        self.channel = channel
        self.queue_name = queue_name
        
    def process_message(self, ch, method, props, body):
        MESSAGES_IN_FLIGHT.labels(queue=self.queue_name).inc()
        
        # Calculate e2e latency from message timestamp
        publish_time = props.timestamp or 0
        e2e_start = time.time()
        
        process_start = time.time()
        status = 'success'
        
        try:
            self._do_work(body)
            ch.basic_ack(method.delivery_tag)
            
        except Exception as e:
            status = 'error'
            ch.basic_nack(method.delivery_tag, requeue=False)
            
        finally:
            process_time = time.time() - process_start
            
            PROCESSING_TIME.labels(
                queue=self.queue_name,
                handler=self._get_handler_name(body)
            ).observe(process_time)
            
            if publish_time:
                e2e_latency = time.time() - publish_time
                MESSAGE_LATENCY.labels(
                    queue=self.queue_name,
                    status=status
                ).observe(e2e_latency)
                
            MESSAGES_PROCESSED.labels(
                queue=self.queue_name,
                status=status
            ).inc()
            
            MESSAGES_IN_FLIGHT.labels(queue=self.queue_name).dec()


# SLA alert rules
"""
# P99 latency > 5s
- alert: MessageLatencyHigh
  expr: histogram_quantile(0.99, rate(message_latency_seconds_bucket[5m])) > 5

# Error rate > 1%
- alert: MessageErrorRateHigh
  expr: rate(messages_processed_total{status="error"}[5m]) / rate(messages_processed_total[5m]) > 0.01

# Throughput dropped > 50%
- alert: ThroughputDropped
  expr: rate(messages_processed_total[5m]) < rate(messages_processed_total[1h] offset 1d) * 0.5
"""
```
