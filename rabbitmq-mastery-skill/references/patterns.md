# Advanced RabbitMQ Patterns

## 1. Saga Pattern Implementation

Distributed transactions across microservices using choreography.

```python
"""
Saga: Order -> Payment -> Inventory -> Shipping
Compensation: Reverse order on any failure
"""

class SagaOrchestrator:
    def __init__(self, channel):
        self.channel = channel
        self.saga_store = {}  # Redis in production
        
    def start_saga(self, saga_id, order_data):
        self.saga_store[saga_id] = {
            'state': 'STARTED',
            'data': order_data,
            'completed_steps': [],
            'started_at': time.time()
        }
        
        # Step 1: Reserve payment
        self.channel.basic_publish(
            exchange='saga.commands',
            routing_key='payment.reserve',
            body=json.dumps({
                'saga_id': saga_id,
                'amount': order_data['total'],
                'customer_id': order_data['customer_id']
            }),
            properties=pika.BasicProperties(
                correlation_id=saga_id,
                reply_to='saga.events',
                headers={'saga_step': 'PAYMENT_RESERVE'}
            )
        )
        
    def handle_event(self, ch, method, props, body):
        event = json.loads(body)
        saga_id = props.correlation_id
        saga = self.saga_store.get(saga_id)
        
        if not saga:
            return
            
        event_type = event.get('type')
        
        if event_type == 'PAYMENT_RESERVED':
            saga['completed_steps'].append('PAYMENT')
            # Step 2: Reserve inventory
            self.channel.basic_publish(
                exchange='saga.commands',
                routing_key='inventory.reserve',
                body=json.dumps({
                    'saga_id': saga_id,
                    'items': saga['data']['items']
                }),
                properties=pika.BasicProperties(
                    correlation_id=saga_id,
                    reply_to='saga.events',
                    headers={'saga_step': 'INVENTORY_RESERVE'}
                )
            )
            
        elif event_type == 'INVENTORY_RESERVED':
            saga['completed_steps'].append('INVENTORY')
            # Step 3: Create shipment
            self.channel.basic_publish(
                exchange='saga.commands',
                routing_key='shipping.create',
                body=json.dumps({
                    'saga_id': saga_id,
                    'address': saga['data']['address'],
                    'items': saga['data']['items']
                }),
                properties=pika.BasicProperties(
                    correlation_id=saga_id,
                    reply_to='saga.events',
                    headers={'saga_step': 'SHIPPING_CREATE'}
                )
            )
            
        elif event_type == 'SHIPMENT_CREATED':
            saga['completed_steps'].append('SHIPPING')
            saga['state'] = 'COMPLETED'
            self._complete_saga(saga_id)
            
        elif event_type.endswith('_FAILED'):
            self._compensate(saga_id, saga)
            
    def _compensate(self, saga_id, saga):
        """Reverse completed steps in reverse order"""
        saga['state'] = 'COMPENSATING'
        
        compensations = {
            'SHIPPING': ('shipping.cancel', 'shipping_id'),
            'INVENTORY': ('inventory.release', 'reservation_id'),
            'PAYMENT': ('payment.release', 'payment_id')
        }
        
        for step in reversed(saga['completed_steps']):
            if step in compensations:
                routing_key, id_field = compensations[step]
                self.channel.basic_publish(
                    exchange='saga.commands',
                    routing_key=routing_key,
                    body=json.dumps({
                        'saga_id': saga_id,
                        id_field: saga['data'].get(id_field)
                    }),
                    properties=pika.BasicProperties(
                        correlation_id=saga_id,
                        headers={'saga_step': f'{step}_COMPENSATE'}
                    )
                )
```

## 2. Event Sourcing with RabbitMQ

```python
"""
Event Store pattern using RabbitMQ Streams
"""

class EventStore:
    def __init__(self, channel):
        self.channel = channel
        self._setup_streams()
        
    def _setup_streams(self):
        # One stream per aggregate type
        for aggregate in ['orders', 'customers', 'products']:
            self.channel.queue_declare(
                f'events.{aggregate}',
                durable=True,
                arguments={
                    'x-queue-type': 'stream',
                    'x-max-age': '365D',  # 1 year retention
                    'x-stream-max-segment-size-bytes': 100_000_000
                }
            )
    
    def append_event(self, aggregate_type, aggregate_id, event_type, data, 
                     expected_version=None):
        event = {
            'aggregate_id': aggregate_id,
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'version': expected_version + 1 if expected_version else 1,
            'event_id': str(uuid4())
        }
        
        self.channel.basic_publish(
            exchange='',
            routing_key=f'events.{aggregate_type}',
            body=json.dumps(event),
            properties=pika.BasicProperties(
                delivery_mode=2,
                message_id=event['event_id'],
                headers={
                    'aggregate_id': aggregate_id,
                    'event_type': event_type,
                    'version': event['version']
                }
            )
        )
        return event
    
    def get_events(self, aggregate_type, aggregate_id, from_version=0):
        """Replay events for an aggregate"""
        events = []
        
        def collector(ch, method, props, body):
            if props.headers.get('aggregate_id') == aggregate_id:
                if props.headers.get('version', 0) > from_version:
                    events.append(json.loads(body))
        
        # Consume from beginning
        self.channel.basic_consume(
            f'events.{aggregate_type}',
            collector,
            arguments={'x-stream-offset': 'first'}
        )
        
        # Process until caught up (with timeout)
        timeout = time.time() + 5
        while time.time() < timeout:
            self.channel.connection.process_data_events(time_limit=0.1)
            
        return sorted(events, key=lambda e: e['version'])


class OrderAggregate:
    """Event-sourced Order aggregate"""
    
    def __init__(self, order_id, events=None):
        self.order_id = order_id
        self.version = 0
        self.status = None
        self.items = []
        self.total = 0
        
        # Replay events
        for event in (events or []):
            self._apply(event)
            
    def _apply(self, event):
        handler = getattr(self, f"_apply_{event['event_type']}", None)
        if handler:
            handler(event['data'])
        self.version = event['version']
        
    def _apply_OrderCreated(self, data):
        self.status = 'CREATED'
        self.items = data['items']
        self.total = data['total']
        
    def _apply_OrderPaid(self, data):
        self.status = 'PAID'
        
    def _apply_OrderShipped(self, data):
        self.status = 'SHIPPED'
        
    def _apply_OrderCancelled(self, data):
        self.status = 'CANCELLED'
```

## 3. CQRS with Separate Read Models

```python
"""
Command Query Responsibility Segregation
Commands -> RabbitMQ -> Event Handlers -> Read Models
"""

# Command side - writes to event stream
class CommandHandler:
    def __init__(self, event_store):
        self.event_store = event_store
        
    def handle_create_order(self, command):
        # Validate
        if not command.get('items'):
            raise ValidationError("Order must have items")
            
        # Create event
        event = self.event_store.append_event(
            'orders',
            str(uuid4()),
            'OrderCreated',
            {
                'customer_id': command['customer_id'],
                'items': command['items'],
                'total': sum(i['price'] * i['qty'] for i in command['items'])
            }
        )
        return event['aggregate_id']


# Query side - maintains denormalized views
class OrderProjection:
    """Updates read models from events"""
    
    def __init__(self, channel, db):
        self.channel = channel
        self.db = db  # PostgreSQL/MongoDB/Redis
        
        # Subscribe to order events
        channel.queue_declare('projection.orders', durable=True)
        channel.queue_bind('projection.orders', 'events.fanout', 'orders.*')
        channel.basic_consume('projection.orders', self.handle_event)
        
    def handle_event(self, ch, method, props, body):
        event = json.loads(body)
        handler = getattr(self, f"handle_{event['event_type']}", None)
        
        if handler:
            try:
                handler(event)
                ch.basic_ack(method.delivery_tag)
            except Exception as e:
                # Retry or DLQ
                ch.basic_nack(method.delivery_tag, requeue=False)
        else:
            ch.basic_ack(method.delivery_tag)
            
    def handle_OrderCreated(self, event):
        self.db.orders.insert_one({
            '_id': event['aggregate_id'],
            'customer_id': event['data']['customer_id'],
            'items': event['data']['items'],
            'total': event['data']['total'],
            'status': 'CREATED',
            'created_at': event['timestamp'],
            'version': event['version']
        })
        
        # Update customer order count (denormalized)
        self.db.customers.update_one(
            {'_id': event['data']['customer_id']},
            {'$inc': {'order_count': 1, 'total_spent': event['data']['total']}}
        )
        
    def handle_OrderShipped(self, event):
        self.db.orders.update_one(
            {'_id': event['aggregate_id']},
            {
                '$set': {
                    'status': 'SHIPPED',
                    'shipped_at': event['timestamp'],
                    'tracking_number': event['data'].get('tracking_number')
                }
            }
        )


# Query service - reads from projections
class OrderQueryService:
    def __init__(self, db):
        self.db = db
        
    def get_order(self, order_id):
        return self.db.orders.find_one({'_id': order_id})
        
    def get_customer_orders(self, customer_id, page=1, limit=20):
        return list(self.db.orders.find(
            {'customer_id': customer_id}
        ).sort('created_at', -1).skip((page-1)*limit).limit(limit))
        
    def get_orders_by_status(self, status):
        return list(self.db.orders.find({'status': status}))
```

## 4. Competing Consumers with Work Stealing

```python
"""
Dynamic load balancing across consumers
"""

class WorkStealingConsumer:
    def __init__(self, channel, queue_name, worker_id):
        self.channel = channel
        self.queue_name = queue_name
        self.worker_id = worker_id
        self.processed = 0
        self.processing_times = []
        
    def start(self):
        # Adaptive prefetch based on processing speed
        initial_prefetch = 10
        self.channel.basic_qos(prefetch_count=initial_prefetch)
        
        self.channel.basic_consume(
            self.queue_name,
            self.process_message,
            consumer_tag=f'worker-{self.worker_id}'
        )
        
        # Start adaptive prefetch adjustment
        self._schedule_prefetch_adjustment()
        
    def process_message(self, ch, method, props, body):
        start = time.time()
        
        try:
            self._do_work(body)
            ch.basic_ack(method.delivery_tag)
            self.processed += 1
            
        except Exception as e:
            ch.basic_nack(method.delivery_tag, requeue=True)
            
        finally:
            elapsed = time.time() - start
            self.processing_times.append(elapsed)
            # Keep last 100 timings
            self.processing_times = self.processing_times[-100:]
            
    def _schedule_prefetch_adjustment(self):
        """Adjust prefetch every 30 seconds based on throughput"""
        def adjust():
            if len(self.processing_times) < 10:
                return
                
            avg_time = sum(self.processing_times) / len(self.processing_times)
            
            # Target: keep ~5 seconds of work buffered
            target_buffer_seconds = 5
            optimal_prefetch = max(1, int(target_buffer_seconds / avg_time))
            optimal_prefetch = min(optimal_prefetch, 200)  # Cap at 200
            
            self.channel.basic_qos(prefetch_count=optimal_prefetch)
            print(f"Worker {self.worker_id}: adjusted prefetch to {optimal_prefetch}")
            
        # Schedule periodic adjustment
        import threading
        timer = threading.Timer(30.0, adjust)
        timer.daemon = True
        timer.start()
```

## 5. Request-Reply Pattern (RPC)

```python
"""
Synchronous RPC over async messaging
"""

class RpcClient:
    def __init__(self, connection):
        self.connection = connection
        self.channel = connection.channel()
        
        # Exclusive callback queue
        result = self.channel.queue_declare('', exclusive=True)
        self.callback_queue = result.method.queue
        
        self.pending = {}
        
        self.channel.basic_consume(
            self.callback_queue,
            self._on_response,
            auto_ack=True
        )
        
    def _on_response(self, ch, method, props, body):
        if props.correlation_id in self.pending:
            self.pending[props.correlation_id] = body
            
    def call(self, routing_key, payload, timeout=30):
        correlation_id = str(uuid4())
        self.pending[correlation_id] = None
        
        self.channel.basic_publish(
            exchange='rpc',
            routing_key=routing_key,
            body=json.dumps(payload),
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=correlation_id,
                expiration=str(timeout * 1000)
            )
        )
        
        # Wait for response
        start = time.time()
        while self.pending[correlation_id] is None:
            if time.time() - start > timeout:
                del self.pending[correlation_id]
                raise TimeoutError(f"RPC call timed out after {timeout}s")
            self.connection.process_data_events(time_limit=0.1)
            
        response = self.pending.pop(correlation_id)
        return json.loads(response)


class RpcServer:
    def __init__(self, channel, queue_name):
        self.channel = channel
        self.handlers = {}
        
        channel.queue_declare(queue_name, durable=True)
        channel.basic_qos(prefetch_count=10)
        channel.basic_consume(queue_name, self._on_request)
        
    def register(self, method_name, handler):
        self.handlers[method_name] = handler
        
    def _on_request(self, ch, method, props, body):
        request = json.loads(body)
        
        try:
            handler = self.handlers.get(request.get('method'))
            if handler:
                result = handler(request.get('params', {}))
                response = {'result': result, 'error': None}
            else:
                response = {'result': None, 'error': 'Method not found'}
                
        except Exception as e:
            response = {'result': None, 'error': str(e)}
            
        # Send response
        ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            body=json.dumps(response),
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            )
        )
        ch.basic_ack(method.delivery_tag)
```

## 6. Consistent Hash Exchange (Sticky Routing)

```python
"""
Route messages consistently to same consumer based on key
Useful for: ordering guarantees, session affinity, caching
"""

# Enable plugin: rabbitmq-plugins enable rabbitmq_consistent_hash_exchange

def setup_consistent_hash():
    channel.exchange_declare(
        'orders.consistent',
        'x-consistent-hash',
        durable=True
    )
    
    # Create worker queues with weights
    for i in range(4):
        queue_name = f'worker.{i}'
        channel.queue_declare(queue_name, durable=True)
        # Weight determines hash ring coverage
        channel.queue_bind(
            queue_name,
            'orders.consistent',
            routing_key='10'  # Weight of 10
        )

def publish_with_affinity(channel, customer_id, order):
    """All orders from same customer go to same worker"""
    channel.basic_publish(
        exchange='orders.consistent',
        routing_key=customer_id,  # Hash key
        body=json.dumps(order),
        properties=pika.BasicProperties(
            delivery_mode=2,
            headers={'customer_id': customer_id}
        )
    )
```

## 7. Message Deduplication

```python
"""
Exactly-once processing semantics
"""

import hashlib
from redis import Redis

class DeduplicatingConsumer:
    def __init__(self, channel, redis_client, dedup_window=3600):
        self.channel = channel
        self.redis = redis_client
        self.dedup_window = dedup_window  # seconds
        
    def _get_dedup_key(self, props, body):
        """Generate deduplication key"""
        # Use message_id if available
        if props.message_id:
            return f"dedup:{props.message_id}"
            
        # Otherwise hash the content
        content_hash = hashlib.sha256(body).hexdigest()
        return f"dedup:{content_hash}"
        
    def process_message(self, ch, method, props, body):
        dedup_key = self._get_dedup_key(props, body)
        
        # Try to acquire lock (atomic check-and-set)
        acquired = self.redis.set(
            dedup_key,
            '1',
            nx=True,  # Only set if not exists
            ex=self.dedup_window
        )
        
        if not acquired:
            # Duplicate - already processed
            print(f"Duplicate message detected: {dedup_key}")
            ch.basic_ack(method.delivery_tag)
            return
            
        try:
            # Process the message
            self._do_work(body)
            ch.basic_ack(method.delivery_tag)
            
        except Exception as e:
            # Remove dedup key on failure to allow retry
            self.redis.delete(dedup_key)
            ch.basic_nack(method.delivery_tag, requeue=True)
```

## 8. Circuit Breaker for Publishers

```python
"""
Prevent cascade failures when RabbitMQ is struggling
"""

from enum import Enum
import threading

class CircuitState(Enum):
    CLOSED = 'closed'      # Normal operation
    OPEN = 'open'          # Failing fast
    HALF_OPEN = 'half_open'  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, 
                 half_open_requests=3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
        
    def can_execute(self):
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
                
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.successes = 0
                    return True
                return False
                
            if self.state == CircuitState.HALF_OPEN:
                return True
                
    def record_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.successes += 1
                if self.successes >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failures = 0
            else:
                self.failures = 0
                
    def record_failure(self):
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN


class ResilientPublisher:
    def __init__(self, channel):
        self.channel = channel
        self.circuit_breaker = CircuitBreaker()
        self.fallback_queue = []  # Local buffer
        
    def publish(self, exchange, routing_key, body, properties=None):
        if not self.circuit_breaker.can_execute():
            # Circuit open - buffer locally
            self.fallback_queue.append({
                'exchange': exchange,
                'routing_key': routing_key,
                'body': body,
                'properties': properties,
                'timestamp': time.time()
            })
            raise CircuitOpenError("Circuit breaker is open")
            
        try:
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=body,
                properties=properties,
                mandatory=True
            )
            self.circuit_breaker.record_success()
            
            # Drain fallback queue on success
            self._drain_fallback_queue()
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
            
    def _drain_fallback_queue(self):
        while self.fallback_queue and self.circuit_breaker.can_execute():
            msg = self.fallback_queue.pop(0)
            try:
                self.channel.basic_publish(
                    exchange=msg['exchange'],
                    routing_key=msg['routing_key'],
                    body=msg['body'],
                    properties=msg['properties']
                )
            except:
                self.fallback_queue.insert(0, msg)
                break
```

## 9. Delayed Retry with Exponential Backoff (No Plugin)

```python
"""
Retry pattern using TTL + DLX chain
delay.5s -> DLX -> delay.30s -> DLX -> delay.5m -> DLX -> main queue
"""

def setup_retry_chain(channel, main_queue):
    delays = [5000, 30000, 300000]  # 5s, 30s, 5min
    
    # Main exchange
    channel.exchange_declare('retry.exchange', 'direct', durable=True)
    
    # Create delay queues
    for i, delay in enumerate(delays):
        delay_queue = f'{main_queue}.delay.{delay}'
        
        # Delay queue routes to main queue after TTL
        channel.queue_declare(
            delay_queue,
            durable=True,
            arguments={
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': main_queue,
                'x-message-ttl': delay
            }
        )
        
        channel.queue_bind(delay_queue, 'retry.exchange', f'delay.{delay}')
    
    # DLQ for exhausted retries
    channel.queue_declare(f'{main_queue}.dlq', durable=True)


def retry_message(channel, method, props, body, main_queue):
    headers = props.headers or {}
    retry_count = headers.get('x-retry-count', 0)
    
    delays = [5000, 30000, 300000]
    
    if retry_count >= len(delays):
        # Exhausted - send to DLQ
        channel.basic_publish(
            exchange='',
            routing_key=f'{main_queue}.dlq',
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                headers={**headers, 'x-retry-count': retry_count, 'x-final-error': 'max_retries'}
            )
        )
    else:
        # Send to appropriate delay queue
        delay = delays[retry_count]
        channel.basic_publish(
            exchange='retry.exchange',
            routing_key=f'delay.{delay}',
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                headers={**headers, 'x-retry-count': retry_count + 1}
            )
        )
    
    channel.basic_ack(method.delivery_tag)
```

## 10. Message Versioning & Schema Evolution

```python
"""
Handle schema changes gracefully
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class OrderCreatedV1:
    order_id: str
    customer_id: str
    total: float

@dataclass  
class OrderCreatedV2:
    order_id: str
    customer_id: str
    total: float
    currency: str = 'USD'  # New field with default
    
@dataclass
class OrderCreatedV3:
    order_id: str
    customer_id: str
    total: float
    currency: str
    tax_amount: Optional[float] = None  # Optional new field


class VersionedConsumer:
    def __init__(self):
        self.handlers = {
            ('OrderCreated', 1): self.handle_order_v1,
            ('OrderCreated', 2): self.handle_order_v2,
            ('OrderCreated', 3): self.handle_order_v3,
        }
        
    def process_message(self, ch, method, props, body):
        headers = props.headers or {}
        event_type = headers.get('event_type')
        version = headers.get('schema_version', 1)
        
        handler = self.handlers.get((event_type, version))
        
        if not handler:
            # Try to upgrade schema
            handler = self._find_compatible_handler(event_type, version)
            
        if handler:
            data = json.loads(body)
            handler(data)
            ch.basic_ack(method.delivery_tag)
        else:
            # Unknown version - DLQ
            ch.basic_nack(method.delivery_tag, requeue=False)
            
    def _find_compatible_handler(self, event_type, version):
        """Find handler that can process older versions"""
        # Try newer handlers that support backward compatibility
        for v in range(version, 10):  # Check up to v10
            handler = self.handlers.get((event_type, v))
            if handler and getattr(handler, 'backward_compatible', False):
                return handler
        return None
        
    def handle_order_v1(self, data):
        # Upgrade to v3 format
        data['currency'] = 'USD'
        data['tax_amount'] = None
        self._process_order(data)
        
    def handle_order_v2(self, data):
        # Upgrade to v3 format
        data['tax_amount'] = None
        self._process_order(data)
        
    def handle_order_v3(self, data):
        self._process_order(data)
    handle_order_v3.backward_compatible = True
```
