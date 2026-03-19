# Library Arsenal - Trusted Dependencies

> "The best code is no code. The second best is someone else's well-tested code."

## Table of Contents
1. [Web Framework](#web-framework)
2. [Database & ORM](#database)
3. [Validation](#validation)
4. [Authentication](#authentication)
5. [Logging & Monitoring](#logging)
6. [Testing](#testing)
7. [Utilities](#utilities)
8. [Security](#security)

---

## Selection Criteria

Sebelum menambahkan dependency, tanyakan:
1. ✅ Apakah actively maintained? (commit dalam 6 bulan terakhir)
2. ✅ Apakah punya TypeScript support native?
3. ✅ Apakah bundle size reasonable?
4. ✅ Apakah ada alternatif built-in di Bun?
5. ✅ Stars > 1000 atau dari maintainer terpercaya?

---

## Web Framework

### Hono (Recommended)
```bash
bun add hono
```

**Why Hono:**
- Ultra-fast (dibuat untuk edge)
- TypeScript-first
- Middleware ecosystem lengkap
- Works perfectly dengan Bun

```typescript
// src/app.ts
import { Hono } from "hono"
import { cors } from "hono/cors"
import { logger } from "hono/logger"
import { secureHeaders } from "hono/secure-headers"
import { timing } from "hono/timing"
import { compress } from "hono/compress"

const app = new Hono()

// Middleware stack
app.use("*", timing())
app.use("*", logger())
app.use("*", secureHeaders())
app.use("*", cors())
app.use("*", compress())

// Routes
app.get("/", (c) => c.text("Hello Bun!"))

// Typed routes
app.get("/users/:id", async (c) => {
  const id = c.req.param("id")
  // TypeScript knows id is string
  return c.json({ id })
})

export { app }
```

### Hono Middleware yang Wajib
```typescript
import { rateLimiter } from "hono-rate-limiter"
import { csrf } from "hono/csrf"
import { etag } from "hono/etag"

// Rate limiting
app.use("/api/*", rateLimiter({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  limit: 100,
  keyGenerator: (c) => c.req.header("x-forwarded-for") || "anonymous",
}))

// CSRF protection
app.use("/api/*", csrf())

// ETag untuk caching
app.use("*", etag())
```

---

## Database

### Drizzle ORM (Recommended)
```bash
bun add drizzle-orm postgres
bun add -d drizzle-kit
```

**Why Drizzle:**
- Type-safe queries
- Migrations built-in
- Zero runtime overhead
- SQL-like syntax

```typescript
// src/db/schema.ts
import { pgTable, text, timestamp, boolean, uuid } from "drizzle-orm/pg-core"

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  email: text("email").notNull().unique(),
  password: text("password").notNull(),
  name: text("name").notNull(),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
})

export const orders = pgTable("orders", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id),
  total: text("total").notNull(),  // Use string for money
  status: text("status").notNull().default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
})

// Relations
export const usersRelations = relations(users, ({ many }) => ({
  orders: many(orders),
}))
```

```typescript
// src/db/index.ts
import { drizzle } from "drizzle-orm/postgres-js"
import postgres from "postgres"
import * as schema from "./schema"
import { env } from "../config/env"

const client = postgres(env.DATABASE_URL, {
  max: 10,  // Connection pool
  idle_timeout: 20,
  connect_timeout: 10,
})

export const db = drizzle(client, { schema })

// Type-safe queries
const user = await db.query.users.findFirst({
  where: eq(users.email, "test@example.com"),
  with: {
    orders: true,  // Include relations
  },
})
```

### Redis (untuk Caching & Sessions)
```bash
bun add ioredis
```

```typescript
// src/config/redis.ts
import Redis from "ioredis"
import { env } from "./env"

export const redis = new Redis(env.REDIS_URL, {
  maxRetriesPerRequest: 3,
  retryDelayOnFailover: 1000,
  lazyConnect: true,
})

// Cache helper
export const cache = {
  async get<T>(key: string): Promise<T | null> {
    const data = await redis.get(key)
    return data ? JSON.parse(data) : null
  },
  
  async set(key: string, value: unknown, ttlSeconds = 3600): Promise<void> {
    await redis.setex(key, ttlSeconds, JSON.stringify(value))
  },
  
  async del(key: string): Promise<void> {
    await redis.del(key)
  },
  
  async invalidatePattern(pattern: string): Promise<void> {
    const keys = await redis.keys(pattern)
    if (keys.length > 0) {
      await redis.del(...keys)
    }
  },
}

// Usage
const user = await cache.get<User>(`user:${id}`)
if (!user) {
  const freshUser = await db.query.users.findFirst({ where: eq(users.id, id) })
  await cache.set(`user:${id}`, freshUser, 300)  // 5 min TTL
}
```

---

## Validation

### Zod (Recommended)
```bash
bun add zod
```

**Why Zod:**
- TypeScript-first
- Zero dependencies
- Composable schemas
- Great error messages

```typescript
// src/schemas/user.schema.ts
import { z } from "zod"

export const createUserSchema = z.object({
  email: z.string().email("Invalid email format"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .regex(/[A-Z]/, "Password must contain uppercase")
    .regex(/[0-9]/, "Password must contain number"),
  name: z.string().min(2).max(100),
  age: z.number().int().min(13).max(120).optional(),
})

export type CreateUserInput = z.infer<typeof createUserSchema>

// Dengan Hono validator
import { zValidator } from "@hono/zod-validator"

app.post("/users", 
  zValidator("json", createUserSchema),
  async (c) => {
    const input = c.req.valid("json")
    // input sudah typed dan validated!
    const user = await userService.create(input)
    return c.json(user, 201)
  }
)
```

### Advanced Zod Patterns
```typescript
// Refinements
const passwordSchema = z.object({
  password: z.string().min(8),
  confirmPassword: z.string(),
}).refine(
  (data) => data.password === data.confirmPassword,
  { message: "Passwords don't match", path: ["confirmPassword"] }
)

// Transform
const dateStringSchema = z.string().transform((str) => new Date(str))

// Coerce (auto-convert types)
const paginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
})

// Discriminated unions
const paymentSchema = z.discriminatedUnion("method", [
  z.object({
    method: z.literal("card"),
    cardNumber: z.string().length(16),
    cvv: z.string().length(3),
  }),
  z.object({
    method: z.literal("bank_transfer"),
    bankCode: z.string(),
    accountNumber: z.string(),
  }),
])
```

---

## Authentication

### Lucia Auth (Recommended)
```bash
bun add lucia @lucia-auth/adapter-drizzle
```

**Why Lucia:**
- Session-based (more secure than JWT for web)
- Database adapter untuk Drizzle
- TypeScript-first
- Actively maintained

```typescript
// src/lib/auth.ts
import { Lucia } from "lucia"
import { DrizzlePostgreSQLAdapter } from "@lucia-auth/adapter-drizzle"
import { db } from "../db"
import { users, sessions } from "../db/schema"

const adapter = new DrizzlePostgreSQLAdapter(db, sessions, users)

export const lucia = new Lucia(adapter, {
  sessionCookie: {
    attributes: {
      secure: process.env.NODE_ENV === "production",
    },
  },
  getUserAttributes: (attributes) => ({
    email: attributes.email,
    name: attributes.name,
  }),
})

declare module "lucia" {
  interface Register {
    Lucia: typeof lucia
    DatabaseUserAttributes: {
      email: string
      name: string
    }
  }
}
```

### Jose (untuk JWT jika diperlukan)
```bash
bun add jose
```

```typescript
// src/utils/jwt.ts
import { SignJWT, jwtVerify } from "jose"
import { env } from "../config/env"

const secret = new TextEncoder().encode(env.JWT_SECRET)

export const jwt = {
  async sign(payload: Record<string, unknown>, expiresIn = "7d") {
    return new SignJWT(payload)
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime(expiresIn)
      .sign(secret)
  },
  
  async verify<T>(token: string): Promise<T> {
    const { payload } = await jwtVerify(token, secret)
    return payload as T
  },
}
```

---

## Logging

### Pino (Recommended)
```bash
bun add pino
bun add -d pino-pretty
```

**Why Pino:**
- Fastest logger for Node/Bun
- JSON structured logging
- Low overhead
- Production-ready

```typescript
// src/utils/logger.ts
import pino from "pino"
import { env } from "../config/env"

export const logger = pino({
  level: env.LOG_LEVEL || "info",
  formatters: {
    level: (label) => ({ level: label }),
  },
  transport: env.NODE_ENV !== "production" 
    ? { target: "pino-pretty" }
    : undefined,
  redact: ["password", "token", "authorization"],
})

// Module-specific loggers
export const dbLogger = logger.child({ module: "database" })
export const authLogger = logger.child({ module: "auth" })
export const apiLogger = logger.child({ module: "api" })
```

---

## Testing

### Vitest (Recommended)
```bash
bun add -d vitest @vitest/coverage-v8
```

**Why Vitest:**
- Fast (Vite-powered)
- Jest-compatible API
- Native TypeScript
- Watch mode yang excellent

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      exclude: ["node_modules", "tests"],
    },
    include: ["tests/**/*.test.ts"],
    setupFiles: ["tests/setup.ts"],
  },
})
```

```typescript
// tests/user.service.test.ts
import { describe, it, expect, beforeEach, vi } from "vitest"
import { userService } from "../src/services/user.service"
import { db } from "../src/db"

// Mock database
vi.mock("../src/db", () => ({
  db: {
    query: {
      users: {
        findFirst: vi.fn(),
      },
    },
    insert: vi.fn().mockReturnValue({
      values: vi.fn().mockReturnValue({
        returning: vi.fn(),
      }),
    }),
  },
}))

describe("UserService", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("should create user with hashed password", async () => {
    const input = { email: "test@example.com", password: "Test123!", name: "Test" }
    
    vi.mocked(db.query.users.findFirst).mockResolvedValue(null)
    vi.mocked(db.insert).mockReturnValue({
      values: vi.fn().mockReturnValue({
        returning: vi.fn().mockResolvedValue([{ id: "1", ...input }]),
      }),
    } as any)
    
    const user = await userService.create(input)
    
    expect(user.email).toBe(input.email)
    expect(user.password).not.toBe(input.password)  // Should be hashed
  })
})
```

### Supertest (untuk API Testing)
```bash
bun add -d supertest
```

```typescript
// tests/api/user.api.test.ts
import { describe, it, expect } from "vitest"
import { app } from "../../src/app"

describe("User API", () => {
  it("POST /users - should create user", async () => {
    const res = await app.request("/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: "test@example.com",
        password: "Test123!",
        name: "Test User",
      }),
    })
    
    expect(res.status).toBe(201)
    const json = await res.json()
    expect(json.success).toBe(true)
    expect(json.data.email).toBe("test@example.com")
  })
  
  it("POST /users - should reject invalid email", async () => {
    const res = await app.request("/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: "invalid",
        password: "Test123!",
        name: "Test",
      }),
    })
    
    expect(res.status).toBe(400)
  })
})
```

---

## Utilities

### Date-fns (untuk Date manipulation)
```bash
bun add date-fns
```

```typescript
import { format, addDays, isAfter, parseISO } from "date-fns"

const formatted = format(new Date(), "yyyy-MM-dd HH:mm:ss")
const nextWeek = addDays(new Date(), 7)
const isExpired = isAfter(new Date(), parseISO(expiryDate))
```

### Nanoid (untuk ID generation)
```bash
bun add nanoid
```

```typescript
import { nanoid, customAlphabet } from "nanoid"

// Default (URL-safe)
const id = nanoid()  // "V1StGXR8_Z5jdHi6B-myT"

// Custom alphabet (readable IDs)
const orderId = customAlphabet("0123456789ABCDEF", 10)
// "4F7A2B9C1E"
```

### Lodash-es (utilities)
```bash
bun add lodash-es
bun add -d @types/lodash-es
```

```typescript
// Import only what you need (tree-shaking)
import { debounce, throttle, groupBy, chunk } from "lodash-es"

const grouped = groupBy(users, "role")
const chunked = chunk(items, 100)  // Untuk batch processing
```

---

## Security

### Helmet Headers (via Hono)
```typescript
import { secureHeaders } from "hono/secure-headers"

app.use("*", secureHeaders({
  contentSecurityPolicy: {
    defaultSrc: ["'self'"],
    scriptSrc: ["'self'"],
  },
  xFrameOptions: "DENY",
  xContentTypeOptions: "nosniff",
}))
```

### CORS Configuration
```typescript
import { cors } from "hono/cors"

app.use("/api/*", cors({
  origin: env.ALLOWED_ORIGINS.split(","),
  allowMethods: ["GET", "POST", "PUT", "DELETE"],
  allowHeaders: ["Content-Type", "Authorization"],
  exposeHeaders: ["X-Request-Id"],
  maxAge: 86400,
  credentials: true,
}))
```

### Bun Native Password Hashing
```typescript
// Bun sudah punya built-in password hashing!
// Tidak perlu bcrypt

// Hash password
const hash = await Bun.password.hash("mypassword", {
  algorithm: "argon2id",  // Most secure
  memoryCost: 65536,      // 64MB
  timeCost: 2,
})

// Verify password
const isValid = await Bun.password.verify("mypassword", hash)
```

---

## Package.json Template

```json
{
  "name": "my-bun-app",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "bun --watch src/index.ts",
    "build": "bun build src/index.ts --outdir dist --target bun",
    "start": "bun dist/index.js",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "lint": "bunx @biomejs/biome check .",
    "format": "bunx @biomejs/biome format --write .",
    "db:generate": "drizzle-kit generate",
    "db:migrate": "drizzle-kit migrate",
    "db:studio": "drizzle-kit studio"
  },
  "dependencies": {
    "hono": "^4.x",
    "drizzle-orm": "^0.35.x",
    "postgres": "^3.x",
    "zod": "^3.x",
    "pino": "^9.x",
    "ioredis": "^5.x",
    "lucia": "^3.x",
    "nanoid": "^5.x"
  },
  "devDependencies": {
    "@types/bun": "latest",
    "vitest": "^2.x",
    "drizzle-kit": "^0.25.x",
    "pino-pretty": "^11.x",
    "@biomejs/biome": "^1.x"
  }
}
```
