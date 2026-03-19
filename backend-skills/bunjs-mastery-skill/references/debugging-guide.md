# Debugging Guide untuk Bun.js

## Table of Contents
1. [Common Crash Points](#common-crash-points)
2. [Memory Leaks](#memory-leaks)
3. [Async Debugging](#async-debugging)
4. [Database Issues](#database-issues)
5. [Docker-specific Issues](#docker-issues)
6. [Performance Profiling](#performance-profiling)
7. [Logging Best Practices](#logging)

---

## Common Crash Points

### 1. Unhandled Promise Rejection
```typescript
// ❌ Crash waiting to happen
app.get("/user/:id", async (c) => {
  const user = await db.query.users.findFirst({
    where: eq(users.id, c.req.param("id"))
  })
  return c.json(user)  // Crash jika user null!
})

// ✅ Fixed
app.get("/user/:id", async (c) => {
  const user = await db.query.users.findFirst({
    where: eq(users.id, c.req.param("id"))
  })
  if (!user) {
    throw new NotFoundError("User")
  }
  return c.json(user)
})
```

### 2. Global Error Handler WAJIB
```typescript
// src/index.ts - SELALU tambahkan ini
process.on("uncaughtException", (err) => {
  logger.fatal({ err }, "Uncaught Exception - shutting down")
  process.exit(1)
})

process.on("unhandledRejection", (reason, promise) => {
  logger.error({ reason, promise }, "Unhandled Rejection")
  // Jangan exit, tapi log untuk investigasi
})
```

### 3. Graceful Shutdown
```typescript
// src/index.ts
const server = Bun.serve({
  port: env.PORT,
  fetch: app.fetch,
})

const shutdown = async (signal: string) => {
  logger.info(`${signal} received, shutting down gracefully`)
  
  // Stop accepting new connections
  server.stop()
  
  // Close database connections
  await db.$client.end()
  
  // Close Redis
  await redis.quit()
  
  logger.info("Graceful shutdown complete")
  process.exit(0)
}

process.on("SIGTERM", () => shutdown("SIGTERM"))
process.on("SIGINT", () => shutdown("SIGINT"))
```

---

## Memory Leaks

### Deteksi Memory Leak
```typescript
// Monitor memory usage
setInterval(() => {
  const used = process.memoryUsage()
  logger.debug({
    heapUsed: `${Math.round(used.heapUsed / 1024 / 1024)}MB`,
    heapTotal: `${Math.round(used.heapTotal / 1024 / 1024)}MB`,
    external: `${Math.round(used.external / 1024 / 1024)}MB`,
    rss: `${Math.round(used.rss / 1024 / 1024)}MB`,
  }, "Memory usage")
}, 30000)  // Every 30 seconds
```

### Common Memory Leak Sources

#### 1. Event Listeners Tidak Dibersihkan
```typescript
// ❌ Memory leak
class UserWatcher {
  constructor(userId: string) {
    eventBus.on("user.updated", this.handleUpdate)
  }
  
  handleUpdate = (user: User) => { ... }
}

// ✅ Cleanup properly
class UserWatcher {
  private cleanup?: () => void
  
  constructor(userId: string) {
    eventBus.on("user.updated", this.handleUpdate)
    this.cleanup = () => eventBus.off("user.updated", this.handleUpdate)
  }
  
  handleUpdate = (user: User) => { ... }
  
  dispose() {
    this.cleanup?.()
  }
}
```

#### 2. SetInterval/SetTimeout Tidak Dibersihkan
```typescript
// ❌ Memory leak di long-running process
function startPolling() {
  setInterval(async () => {
    await checkForUpdates()
  }, 5000)
}

// ✅ Track dan clear
const intervals: Timer[] = []

function startPolling() {
  const id = setInterval(async () => {
    await checkForUpdates()
  }, 5000)
  intervals.push(id)
}

function cleanup() {
  intervals.forEach(clearInterval)
  intervals.length = 0
}
```

#### 3. Closure Memory Trap
```typescript
// ❌ Large array tetap di memory karena closure
function createHandler() {
  const hugeData = loadHugeDataset()  // 100MB
  
  return (req: Request) => {
    // hugeData masih referenced!
    return new Response("OK")
  }
}

// ✅ Extract hanya yang dibutuhkan
function createHandler() {
  const hugeData = loadHugeDataset()
  const summary = extractSummary(hugeData)  // 1KB
  // hugeData bisa di-GC
  
  return (req: Request) => {
    return new Response(JSON.stringify(summary))
  }
}
```

---

## Async Debugging

### 1. Race Condition
```typescript
// ❌ Race condition - balance bisa negatif
async function withdraw(userId: string, amount: number) {
  const user = await getUser(userId)
  if (user.balance >= amount) {
    await updateBalance(userId, user.balance - amount)
  }
}

// ✅ Fix dengan database lock
async function withdraw(userId: string, amount: number) {
  return db.transaction(async (tx) => {
    const user = await tx.query.users.findFirst({
      where: eq(users.id, userId),
      // SELECT ... FOR UPDATE (row lock)
    })
    
    if (!user || user.balance < amount) {
      throw new ValidationError("Insufficient balance")
    }
    
    await tx.update(users)
      .set({ balance: user.balance - amount })
      .where(eq(users.id, userId))
  })
}
```

### 2. Deadlock Detection
```typescript
// Timeout wrapper untuk detect deadlock
async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  operation: string
): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => {
      reject(new Error(`${operation} timed out after ${ms}ms`))
    }, ms)
  })
  
  return Promise.race([promise, timeout])
}

// Usage
const user = await withTimeout(
  db.query.users.findFirst({ where: eq(users.id, id) }),
  5000,
  "getUserById"
)
```

### 3. Promise.all Error Handling
```typescript
// ❌ One failure kills all
const [users, orders, products] = await Promise.all([
  getUsers(),
  getOrders(),
  getProducts(),  // Jika ini error, semua hilang
])

// ✅ Use Promise.allSettled untuk partial success
const results = await Promise.allSettled([
  getUsers(),
  getOrders(),
  getProducts(),
])

const [usersResult, ordersResult, productsResult] = results

const users = usersResult.status === "fulfilled" 
  ? usersResult.value 
  : []

const orders = ordersResult.status === "fulfilled"
  ? ordersResult.value
  : []

// Log failures
results
  .filter((r): r is PromiseRejectedResult => r.status === "rejected")
  .forEach(r => logger.error({ reason: r.reason }, "Parallel fetch failed"))
```

---

## Database Issues

### 1. Connection Pool Exhaustion
```typescript
// ❌ Connection leak
async function getUser(id: string) {
  const conn = await pool.connect()
  const result = await conn.query("SELECT * FROM users WHERE id = $1", [id])
  // FORGOT: conn.release()
  return result.rows[0]
}

// ✅ Always release dengan finally
async function getUser(id: string) {
  const conn = await pool.connect()
  try {
    const result = await conn.query("SELECT * FROM users WHERE id = $1", [id])
    return result.rows[0]
  } finally {
    conn.release()  // ALWAYS release
  }
}

// ✅ Better: Use Drizzle yang handle otomatis
const user = await db.query.users.findFirst({
  where: eq(users.id, id)
})
```

### 2. N+1 Query Problem
```typescript
// ❌ N+1 queries
async function getUsersWithOrders() {
  const users = await db.query.users.findMany()
  
  // 1 query untuk users + N queries untuk orders
  return Promise.all(users.map(async (user) => ({
    ...user,
    orders: await db.query.orders.findMany({
      where: eq(orders.userId, user.id)
    })
  })))
}

// ✅ Single query dengan join
async function getUsersWithOrders() {
  return db.query.users.findMany({
    with: {
      orders: true  // Drizzle handles the join
    }
  })
}
```

### 3. Transaction Debugging
```typescript
// Debug transaction issues
async function transferMoney(from: string, to: string, amount: number) {
  logger.debug({ from, to, amount }, "Starting transfer")
  
  try {
    await db.transaction(async (tx) => {
      logger.debug("Transaction started")
      
      const sender = await tx.query.accounts.findFirst({
        where: eq(accounts.id, from)
      })
      logger.debug({ sender }, "Sender fetched")
      
      if (!sender || sender.balance < amount) {
        throw new ValidationError("Insufficient funds")
      }
      
      await tx.update(accounts)
        .set({ balance: sender.balance - amount })
        .where(eq(accounts.id, from))
      logger.debug("Sender balance updated")
      
      await tx.update(accounts)
        .set({ balance: sql`${accounts.balance} + ${amount}` })
        .where(eq(accounts.id, to))
      logger.debug("Receiver balance updated")
    })
    
    logger.info({ from, to, amount }, "Transfer completed")
  } catch (err) {
    logger.error({ err, from, to, amount }, "Transfer failed")
    throw err
  }
}
```

---

## Docker Issues

### 1. Container Crash Loop
```yaml
# docker-compose.yml - tambahkan healthcheck
services:
  api:
    build: .
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
```

### 2. DNS Resolution Issues
```typescript
// Di dalam container, DNS resolution bisa lambat
// ❌ Timeout terlalu pendek
const client = new Redis({
  host: "redis",
  connectTimeout: 1000,  // Terlalu pendek untuk DNS resolution
})

// ✅ Give more time for container DNS
const client = new Redis({
  host: "redis",
  connectTimeout: 10000,
  retryDelayOnFailover: 1000,
  maxRetriesPerRequest: 3,
})
```

### 3. Volume Mount Issues
```yaml
# ❌ Anonymous volume - data hilang saat recreate
volumes:
  - /app/data

# ✅ Named volume - persistent
volumes:
  - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Performance Profiling

### Bun Built-in Profiling
```bash
# CPU profiling
bun --cpu-profile src/index.ts

# Generate flame graph
bun --cpu-profile=profile.json src/index.ts
# View dengan speedscope: https://speedscope.app
```

### Request Timing Middleware
```typescript
// src/middlewares/timing.middleware.ts
import { MiddlewareHandler } from "hono"
import { logger } from "../utils/logger"

export const timingMiddleware: MiddlewareHandler = async (c, next) => {
  const start = performance.now()
  const requestId = crypto.randomUUID()
  
  c.set("requestId", requestId)
  
  await next()
  
  const duration = performance.now() - start
  const status = c.res.status
  
  const logLevel = duration > 1000 ? "warn" : "info"
  
  logger[logLevel]({
    requestId,
    method: c.req.method,
    path: c.req.path,
    status,
    duration: `${duration.toFixed(2)}ms`,
  }, "Request completed")
}
```

### Database Query Timing
```typescript
// Dengan Drizzle logger
const db = drizzle(client, {
  logger: {
    logQuery: (query, params) => {
      logger.debug({ query, params }, "SQL Query")
    },
  },
})

// Custom timing wrapper
async function timedQuery<T>(
  name: string,
  queryFn: () => Promise<T>
): Promise<T> {
  const start = performance.now()
  try {
    return await queryFn()
  } finally {
    const duration = performance.now() - start
    if (duration > 100) {
      logger.warn({ name, duration: `${duration.toFixed(2)}ms` }, "Slow query")
    }
  }
}

// Usage
const users = await timedQuery("getActiveUsers", () =>
  db.query.users.findMany({ where: eq(users.active, true) })
)
```

---

## Logging

### Structured Logging dengan Pino
```typescript
// src/utils/logger.ts
import pino from "pino"
import { env } from "../config/env"

export const logger = pino({
  level: env.NODE_ENV === "production" ? "info" : "debug",
  
  // Pretty print untuk development
  transport: env.NODE_ENV !== "production" ? {
    target: "pino-pretty",
    options: {
      colorize: true,
      translateTime: "SYS:standard",
      ignore: "pid,hostname",
    },
  } : undefined,
  
  // Redact sensitive data
  redact: {
    paths: ["password", "token", "authorization", "cookie"],
    censor: "[REDACTED]",
  },
  
  // Base fields
  base: {
    env: env.NODE_ENV,
    version: process.env.npm_package_version,
  },
})

// Child logger untuk specific module
export const createLogger = (module: string) => 
  logger.child({ module })
```

### Log Levels Best Practice
```typescript
// FATAL: App crash, immediate attention
logger.fatal({ err }, "Database connection lost - shutting down")

// ERROR: Something failed but app continues
logger.error({ err, userId }, "Payment processing failed")

// WARN: Unexpected but handled
logger.warn({ duration: "5000ms" }, "Slow query detected")

// INFO: Important business events
logger.info({ userId, orderId }, "Order placed successfully")

// DEBUG: Detailed technical info (dev only)
logger.debug({ query, params }, "Executing SQL query")

// TRACE: Very detailed (rarely used)
logger.trace({ headers }, "Request headers")
```

### Contextual Logging
```typescript
// Tambahkan request context ke semua logs
import { AsyncLocalStorage } from "node:async_hooks"

interface RequestContext {
  requestId: string
  userId?: string
  path: string
}

export const requestContext = new AsyncLocalStorage<RequestContext>()

// Middleware
app.use(async (c, next) => {
  const context: RequestContext = {
    requestId: crypto.randomUUID(),
    path: c.req.path,
  }
  
  return requestContext.run(context, () => next())
})

// Logger wrapper
export const log = {
  info: (msg: string, data?: object) => {
    const ctx = requestContext.getStore()
    logger.info({ ...ctx, ...data }, msg)
  },
  // ... other levels
}

// Usage
log.info("Processing payment", { amount: 100 })
// Output: {"requestId":"abc-123","path":"/pay","amount":100,"msg":"Processing payment"}
```
