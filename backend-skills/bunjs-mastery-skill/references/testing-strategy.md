# Testing Strategy untuk Bun.js

## Table of Contents
1. [Testing Pyramid](#testing-pyramid)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [E2E Testing](#e2e-testing)
5. [Test Utilities](#test-utilities)
6. [CI Configuration](#ci-configuration)

---

## Testing Pyramid

```
        /\
       /  \      E2E Tests (10%)
      /----\     - Full flow testing
     /      \    - Browser/API simulation
    /--------\
   /          \  Integration Tests (20%)
  /   Logic    \ - Database queries
 /--------------\- External services
/                \
/  Unit Tests     \ Unit Tests (70%)
/   (70%)          \- Pure functions
/==================\ - Business logic
```

**Golden Rule:** Jika test butuh lebih dari 1 second untuk run, itu bukan unit test.

---

## Unit Testing

### Setup Vitest
```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config"
import path from "path"

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    exclude: ["tests/e2e/**/*"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      exclude: [
        "node_modules",
        "tests",
        "**/*.d.ts",
        "**/*.config.ts",
        "src/types/**",
      ],
      thresholds: {
        statements: 80,
        branches: 80,
        functions: 80,
        lines: 80,
      },
    },
    setupFiles: ["tests/setup.ts"],
    testTimeout: 5000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
```

### Test Setup
```typescript
// tests/setup.ts
import { beforeAll, afterAll, afterEach } from "vitest"

// Global setup
beforeAll(() => {
  // Set test environment variables
  process.env.NODE_ENV = "test"
  process.env.LOG_LEVEL = "silent"
})

afterAll(() => {
  // Cleanup
})

afterEach(() => {
  // Reset mocks
})
```

### Pure Function Testing
```typescript
// src/utils/string.util.ts
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
}

export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength - 3) + "..."
}
```

```typescript
// tests/unit/utils/string.util.test.ts
import { describe, it, expect } from "vitest"
import { slugify, truncate } from "@/utils/string.util"

describe("slugify", () => {
  it("should convert text to lowercase", () => {
    expect(slugify("Hello World")).toBe("hello-world")
  })

  it("should replace spaces with hyphens", () => {
    expect(slugify("hello world")).toBe("hello-world")
  })

  it("should remove special characters", () => {
    expect(slugify("hello! @world#")).toBe("hello-world")
  })

  it("should trim leading/trailing hyphens", () => {
    expect(slugify("--hello world--")).toBe("hello-world")
  })

  it("should handle empty string", () => {
    expect(slugify("")).toBe("")
  })

  it("should handle unicode characters", () => {
    expect(slugify("Héllo Wörld")).toBe("hllo-wrld")
  })
})

describe("truncate", () => {
  it("should not truncate short text", () => {
    expect(truncate("hello", 10)).toBe("hello")
  })

  it("should truncate long text with ellipsis", () => {
    expect(truncate("hello world", 8)).toBe("hello...")
  })

  it("should handle exact length", () => {
    expect(truncate("hello", 5)).toBe("hello")
  })
})
```

### Service Layer Testing (with Mocks)
```typescript
// src/services/user.service.ts
import { db } from "@/db"
import { users } from "@/db/schema"
import { eq } from "drizzle-orm"
import { NotFoundError, ConflictError } from "@/utils/errors"

export const userService = {
  async getById(id: string) {
    const user = await db.query.users.findFirst({
      where: eq(users.id, id),
    })
    if (!user) throw new NotFoundError("User", id)
    return user
  },

  async create(data: { email: string; name: string; password: string }) {
    const existing = await db.query.users.findFirst({
      where: eq(users.email, data.email),
    })
    if (existing) throw new ConflictError("Email already exists")

    const hashed = await Bun.password.hash(data.password)
    const [user] = await db
      .insert(users)
      .values({ ...data, password: hashed })
      .returning()
    return user
  },
}
```

```typescript
// tests/unit/services/user.service.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest"
import { userService } from "@/services/user.service"
import { NotFoundError, ConflictError } from "@/utils/errors"

// Mock database
vi.mock("@/db", () => ({
  db: {
    query: {
      users: {
        findFirst: vi.fn(),
      },
    },
    insert: vi.fn(),
  },
}))

// Mock Bun.password
vi.mock("bun", () => ({
  password: {
    hash: vi.fn().mockResolvedValue("hashed_password"),
  },
}))

import { db } from "@/db"

describe("UserService", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe("getById", () => {
    it("should return user when found", async () => {
      const mockUser = { id: "1", email: "test@test.com", name: "Test" }
      vi.mocked(db.query.users.findFirst).mockResolvedValue(mockUser)

      const result = await userService.getById("1")

      expect(result).toEqual(mockUser)
      expect(db.query.users.findFirst).toHaveBeenCalledOnce()
    })

    it("should throw NotFoundError when user not found", async () => {
      vi.mocked(db.query.users.findFirst).mockResolvedValue(undefined)

      await expect(userService.getById("999")).rejects.toThrow(NotFoundError)
    })
  })

  describe("create", () => {
    it("should create user with hashed password", async () => {
      const input = { email: "new@test.com", name: "New", password: "pass123" }
      const mockUser = { id: "1", ...input, password: "hashed_password" }

      vi.mocked(db.query.users.findFirst).mockResolvedValue(undefined)
      vi.mocked(db.insert).mockReturnValue({
        values: vi.fn().mockReturnValue({
          returning: vi.fn().mockResolvedValue([mockUser]),
        }),
      } as any)

      const result = await userService.create(input)

      expect(result.password).toBe("hashed_password")
      expect(result.password).not.toBe(input.password)
    })

    it("should throw ConflictError for duplicate email", async () => {
      const existingUser = { id: "1", email: "exists@test.com" }
      vi.mocked(db.query.users.findFirst).mockResolvedValue(existingUser as any)

      await expect(
        userService.create({
          email: "exists@test.com",
          name: "Test",
          password: "pass123",
        })
      ).rejects.toThrow(ConflictError)
    })
  })
})
```

---

## Integration Testing

### Database Integration Tests
```typescript
// tests/integration/setup.ts
import { beforeAll, afterAll, beforeEach } from "vitest"
import { migrate } from "drizzle-orm/postgres-js/migrator"
import { db, client } from "@/db"

beforeAll(async () => {
  // Run migrations
  await migrate(db, { migrationsFolder: "./drizzle" })
})

beforeEach(async () => {
  // Clean tables before each test
  await db.delete(users)
  await db.delete(orders)
})

afterAll(async () => {
  await client.end()
})
```

```typescript
// tests/integration/user.repository.test.ts
import { describe, it, expect, beforeEach } from "vitest"
import { db } from "@/db"
import { users } from "@/db/schema"
import { eq } from "drizzle-orm"

describe("User Repository Integration", () => {
  beforeEach(async () => {
    // Seed test data
    await db.insert(users).values([
      { id: "1", email: "user1@test.com", name: "User 1", password: "hash" },
      { id: "2", email: "user2@test.com", name: "User 2", password: "hash" },
    ])
  })

  it("should find user by email", async () => {
    const user = await db.query.users.findFirst({
      where: eq(users.email, "user1@test.com"),
    })

    expect(user).toBeDefined()
    expect(user?.name).toBe("User 1")
  })

  it("should create user with all fields", async () => {
    const [newUser] = await db
      .insert(users)
      .values({
        email: "new@test.com",
        name: "New User",
        password: "hashedpass",
      })
      .returning()

    expect(newUser.id).toBeDefined()
    expect(newUser.createdAt).toBeDefined()
  })

  it("should update user", async () => {
    await db.update(users).set({ name: "Updated" }).where(eq(users.id, "1"))

    const updated = await db.query.users.findFirst({
      where: eq(users.id, "1"),
    })

    expect(updated?.name).toBe("Updated")
  })
})
```

### API Integration Tests
```typescript
// tests/integration/api/user.api.test.ts
import { describe, it, expect, beforeEach } from "vitest"
import { app } from "@/app"
import { db } from "@/db"
import { users } from "@/db/schema"

describe("User API Integration", () => {
  beforeEach(async () => {
    await db.delete(users)
  })

  describe("POST /api/users", () => {
    it("should create user and return 201", async () => {
      const res = await app.request("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: "test@test.com",
          name: "Test User",
          password: "Password123!",
        }),
      })

      expect(res.status).toBe(201)

      const json = await res.json()
      expect(json.success).toBe(true)
      expect(json.data.email).toBe("test@test.com")
      expect(json.data.password).toBeUndefined() // Should not return password
    })

    it("should return 400 for invalid email", async () => {
      const res = await app.request("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: "invalid",
          name: "Test",
          password: "Password123!",
        }),
      })

      expect(res.status).toBe(400)
    })

    it("should return 409 for duplicate email", async () => {
      // Create first user
      await db.insert(users).values({
        email: "exists@test.com",
        name: "Existing",
        password: "hash",
      })

      const res = await app.request("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: "exists@test.com",
          name: "New",
          password: "Password123!",
        }),
      })

      expect(res.status).toBe(409)
    })
  })

  describe("GET /api/users/:id", () => {
    it("should return user when exists", async () => {
      const [created] = await db
        .insert(users)
        .values({ email: "test@test.com", name: "Test", password: "hash" })
        .returning()

      const res = await app.request(`/api/users/${created.id}`)

      expect(res.status).toBe(200)
      const json = await res.json()
      expect(json.data.id).toBe(created.id)
    })

    it("should return 404 when not found", async () => {
      const res = await app.request("/api/users/non-existent-id")

      expect(res.status).toBe(404)
    })
  })
})
```

---

## E2E Testing

### Playwright Setup
```bash
bun add -d @playwright/test
bunx playwright install
```

```typescript
// playwright.config.ts
import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  webServer: {
    command: "bun run start",
    url: "http://localhost:3000/health",
    reuseExistingServer: !process.env.CI,
  },
})
```

```typescript
// tests/e2e/user-flow.test.ts
import { test, expect } from "@playwright/test"

test.describe("User Registration Flow", () => {
  test("should register new user via API", async ({ request }) => {
    const response = await request.post("/api/users", {
      data: {
        email: `e2e-${Date.now()}@test.com`,
        name: "E2E Test User",
        password: "TestPass123!",
      },
    })

    expect(response.status()).toBe(201)
    const json = await response.json()
    expect(json.success).toBe(true)
  })

  test("should login and access protected resource", async ({ request }) => {
    // Register
    const email = `e2e-${Date.now()}@test.com`
    await request.post("/api/users", {
      data: {
        email,
        name: "E2E User",
        password: "TestPass123!",
      },
    })

    // Login
    const loginRes = await request.post("/api/auth/login", {
      data: { email, password: "TestPass123!" },
    })
    expect(loginRes.status()).toBe(200)

    const { token } = await loginRes.json()

    // Access protected route
    const protectedRes = await request.get("/api/me", {
      headers: { Authorization: `Bearer ${token}` },
    })
    expect(protectedRes.status()).toBe(200)
  })
})
```

---

## Test Utilities

### Factory Functions
```typescript
// tests/factories/user.factory.ts
import { faker } from "@faker-js/faker"
import type { NewUser } from "@/db/schema"

export function createUserData(overrides: Partial<NewUser> = {}): NewUser {
  return {
    email: faker.internet.email(),
    name: faker.person.fullName(),
    password: faker.internet.password({ length: 12 }),
    ...overrides,
  }
}

export function createManyUsers(count: number, overrides: Partial<NewUser> = []) {
  return Array.from({ length: count }, (_, i) =>
    createUserData(overrides[i] || {})
  )
}
```

### Custom Matchers
```typescript
// tests/matchers/api.matchers.ts
import { expect } from "vitest"

expect.extend({
  toBeSuccessResponse(received) {
    const pass =
      received.success === true && received.data !== undefined

    return {
      pass,
      message: () =>
        pass
          ? `Expected response not to be success`
          : `Expected response to be success, got: ${JSON.stringify(received)}`,
    }
  },

  toBeErrorResponse(received, code: string) {
    const pass =
      received.success === false &&
      received.error?.code === code

    return {
      pass,
      message: () =>
        pass
          ? `Expected response not to be error with code ${code}`
          : `Expected error with code ${code}, got: ${JSON.stringify(received)}`,
    }
  },
})

// Usage
expect(response).toBeSuccessResponse()
expect(response).toBeErrorResponse("NOT_FOUND")
```

### Test Database Helper
```typescript
// tests/helpers/db.helper.ts
import { db } from "@/db"
import { users, orders } from "@/db/schema"

export async function cleanDatabase() {
  // Delete in order of foreign key dependencies
  await db.delete(orders)
  await db.delete(users)
}

export async function seedUsers(count = 5) {
  const data = createManyUsers(count)
  return db.insert(users).values(data).returning()
}
```

---

## CI Configuration

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: oven-sh/setup-bun@v1
        with:
          bun-version: latest
      
      - run: bun install
      - run: bun run test:unit
      - run: bun run test:coverage
      
      - uses: codecov/codecov-action@v4
        with:
          files: ./coverage/lcov.info

  integration-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      
      - uses: oven-sh/setup-bun@v1
      
      - run: bun install
      
      - name: Run migrations
        run: bun run db:migrate
        env:
          DATABASE_URL: postgres://test:test@localhost:5432/test_db
      
      - name: Run integration tests
        run: bun run test:integration
        env:
          DATABASE_URL: postgres://test:test@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379
```

### Package.json Scripts
```json
{
  "scripts": {
    "test": "vitest",
    "test:unit": "vitest run --dir tests/unit",
    "test:integration": "vitest run --dir tests/integration",
    "test:e2e": "playwright test",
    "test:coverage": "vitest run --coverage",
    "test:watch": "vitest --watch"
  }
}
```
