#!/usr/bin/env python3
"""
Production-grade RabbitMQ Connection Pool

Features:
- Thread-safe connection management
- Automatic reconnection
- Health checking
- Channel pooling
- Graceful shutdown

Usage:
    pool = RabbitMQPool(
        host='rabbitmq.prod',
        credentials=('user', 'pass'),
        pool_size=10
    )
    
    with pool.channel() as ch:
        ch.basic_publish(...)
"""

import pika
import threading
import time
import logging
from queue import Queue, Empty
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    host: str = 'localhost'
    port: int = 5672
    vhost: str = '/'
    username: str = 'guest'
    password: str = 'guest'
    
    # Pool settings
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout: float = 30.0
    
    # Connection settings
    heartbeat: int = 60
    blocked_connection_timeout: int = 300
    connection_attempts: int = 3
    retry_delay: float = 5.0
    socket_timeout: float = 10.0
    
    # Health check
    health_check_interval: float = 30.0


class PooledConnection:
    """Wrapper around pika connection with metadata"""
    
    def __init__(self, connection: pika.BlockingConnection, pool: 'RabbitMQPool'):
        self.connection = connection
        self.pool = pool
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self._channel: Optional[pika.channel.Channel] = None
        
    @property
    def channel(self) -> pika.channel.Channel:
        if self._channel is None or self._channel.is_closed:
            self._channel = self.connection.channel()
        return self._channel
        
    def is_healthy(self) -> bool:
        try:
            return self.connection.is_open
        except Exception:
            return False
            
    def close(self):
        try:
            if self._channel and self._channel.is_open:
                self._channel.close()
            if self.connection.is_open:
                self.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")


class RabbitMQPool:
    """Thread-safe RabbitMQ connection pool"""
    
    def __init__(self, **kwargs):
        self.config = PoolConfig(**kwargs)
        self._pool: Queue[PooledConnection] = Queue(maxsize=self.config.pool_size)
        self._overflow_count = 0
        self._lock = threading.Lock()
        self._closed = False
        
        # Connection parameters
        self._params = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            virtual_host=self.config.vhost,
            credentials=pika.PlainCredentials(
                self.config.username,
                self.config.password
            ),
            heartbeat=self.config.heartbeat,
            blocked_connection_timeout=self.config.blocked_connection_timeout,
            connection_attempts=self.config.connection_attempts,
            retry_delay=self.config.retry_delay,
            socket_timeout=self.config.socket_timeout,
            stack_timeout=self.config.socket_timeout + 5,
        )
        
        # Initialize pool
        self._initialize_pool()
        
        # Start health checker
        self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_thread.start()
        
        # Register cleanup
        atexit.register(self.close)
        
    def _initialize_pool(self):
        """Pre-create connections"""
        for _ in range(self.config.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put_nowait(conn)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
                
    def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection"""
        connection = pika.BlockingConnection(self._params)
        return PooledConnection(connection, self)
        
    def _health_check_loop(self):
        """Background health checker"""
        while not self._closed:
            time.sleep(self.config.health_check_interval)
            self._check_pool_health()
            
    def _check_pool_health(self):
        """Check and replace unhealthy connections"""
        healthy = []
        
        # Drain pool
        while True:
            try:
                conn = self._pool.get_nowait()
                if conn.is_healthy():
                    healthy.append(conn)
                else:
                    logger.info("Removing unhealthy connection from pool")
                    conn.close()
            except Empty:
                break
                
        # Refill pool
        for conn in healthy:
            self._pool.put_nowait(conn)
            
        # Create new connections if needed
        current_size = self._pool.qsize()
        if current_size < self.config.pool_size:
            for _ in range(self.config.pool_size - current_size):
                try:
                    conn = self._create_connection()
                    self._pool.put_nowait(conn)
                    logger.info("Added new connection to pool")
                except Exception as e:
                    logger.warning(f"Failed to create replacement connection: {e}")
                    
    def acquire(self, timeout: Optional[float] = None) -> PooledConnection:
        """Acquire a connection from the pool"""
        if self._closed:
            raise RuntimeError("Pool is closed")
            
        timeout = timeout or self.config.pool_timeout
        
        try:
            conn = self._pool.get(timeout=timeout)
            
            # Verify connection is healthy
            if not conn.is_healthy():
                conn.close()
                conn = self._create_connection()
                
            conn.last_used = time.time()
            conn.use_count += 1
            return conn
            
        except Empty:
            # Try to create overflow connection
            with self._lock:
                if self._overflow_count < self.config.max_overflow:
                    self._overflow_count += 1
                    logger.info(f"Creating overflow connection ({self._overflow_count})")
                    return self._create_connection()
                    
            raise TimeoutError(f"Could not acquire connection within {timeout}s")
            
    def release(self, conn: PooledConnection):
        """Return a connection to the pool"""
        if self._closed:
            conn.close()
            return
            
        # Check if this was an overflow connection
        with self._lock:
            if self._pool.full():
                self._overflow_count = max(0, self._overflow_count - 1)
                conn.close()
                return
                
        # Return to pool if healthy
        if conn.is_healthy():
            try:
                self._pool.put_nowait(conn)
            except:
                conn.close()
        else:
            conn.close()
            
    @contextmanager
    def connection(self):
        """Context manager for acquiring a connection"""
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)
            
    @contextmanager
    def channel(self):
        """Context manager for acquiring a channel"""
        conn = self.acquire()
        try:
            yield conn.channel
        finally:
            self.release(conn)
            
    def close(self):
        """Close all connections and shutdown pool"""
        self._closed = True
        
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
                
        logger.info("Connection pool closed")
        
    @property
    def size(self) -> int:
        return self._pool.qsize()
        
    @property
    def stats(self) -> dict:
        return {
            'pool_size': self._pool.qsize(),
            'max_size': self.config.pool_size,
            'overflow': self._overflow_count,
            'max_overflow': self.config.max_overflow,
        }


# Convenience function
def create_pool(**kwargs) -> RabbitMQPool:
    """Create a connection pool with sensible defaults"""
    return RabbitMQPool(**kwargs)


if __name__ == '__main__':
    # Example usage
    pool = create_pool(
        host='localhost',
        username='guest',
        password='guest',
        pool_size=5
    )
    
    print(f"Pool stats: {pool.stats}")
    
    # Use connection
    with pool.channel() as ch:
        ch.queue_declare('test_queue', durable=True)
        ch.basic_publish(
            exchange='',
            routing_key='test_queue',
            body=b'Hello from pool!',
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print("Message published!")
        
    print(f"Pool stats after use: {pool.stats}")
    
    pool.close()
